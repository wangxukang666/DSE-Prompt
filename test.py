import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
from FAPrompt import FAPrompt
from dataset import Dataset
from logger import get_logger
import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
from scipy.ndimage import gaussian_filter
import math
from metrics import image_level_metrics, pixel_level_metrics 
from tqdm import tqdm

class SCA_Module(nn.Module):
    def __init__(self, embed_dim=768, num_anchors=4, num_patches=1369):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_anchors = num_anchors
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.anchor_queries = nn.Parameter(torch.zeros(num_anchors, embed_dim))
        nn.init.trunc_normal_(self.anchor_queries, std=0.02)
        self.gate_w = nn.Linear(embed_dim, num_anchors, bias=False)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, text_features, patch_features, threshold=None, ext_mask=None, fallback_penalty=0.1):
        B, N, D = patch_features.shape
        P_pos = patch_features + self.pos_embed
        Q_anchor = self.anchor_queries.unsqueeze(0).expand(B, -1, -1)
        
        attn_logits = torch.matmul(Q_anchor, P_pos.transpose(1, 2)) / math.sqrt(self.embed_dim)
        penalty_mask = torch.ones((B, self.num_anchors, 1), device=patch_features.device)
        
        if threshold is not None or ext_mask is not None:
            mask = torch.ones_like(attn_logits, dtype=torch.bool)
            
            if threshold is not None:
                Q_norm = F.normalize(Q_anchor, dim=-1, eps=1e-8)
                P_pos_norm = F.normalize(P_pos, dim=-1, eps=1e-8)
                cos_sim = torch.matmul(Q_norm, P_pos_norm.transpose(1, 2)) 
                
                thresh_mask = cos_sim >= threshold
                valid_counts = thresh_mask.sum(dim=-1, keepdim=True)
                no_valid = (valid_counts == 0)
                
                if no_valid.any():
                    _, max_indices = torch.max(cos_sim, dim=-1, keepdim=True)
                    fallback_mask = torch.zeros_like(thresh_mask).scatter_(-1, max_indices, True)
                    thresh_mask = torch.where(no_valid.expand_as(thresh_mask), fallback_mask, thresh_mask)
                    penalty_mask = torch.where(no_valid, torch.tensor(fallback_penalty, device=patch_features.device), penalty_mask)
                    
                mask = mask & thresh_mask
                
            if ext_mask is not None:
                mask = mask & ext_mask.unsqueeze(1)
                ext_valid_counts = mask.sum(dim=-1, keepdim=True)
                ext_no_valid = (ext_valid_counts == 0)
                if ext_no_valid.any():
                    _, max_idx = torch.max(attn_logits, dim=-1, keepdim=True)
                    fb_mask = torch.zeros_like(mask).scatter_(-1, max_idx, True)
                    mask = torch.where(ext_no_valid.expand_as(mask), fb_mask, mask)
                    
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
            
        attn = F.softmax(attn_logits, dim=-1) 
        attn = attn * penalty_mask
        
        U = torch.matmul(attn, patch_features) 
        gate = torch.sigmoid(self.gate_w(text_features)) 
        delta_t = torch.matmul(gate, U) 
        enhanced_text_features = text_features + self.res_scale * delta_t
        
        return F.normalize(enhanced_text_features, dim=-1, eps=1e-8)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(args):
    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list
    results = {obj: {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': []} for obj in obj_list}

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    prompt_learner = FAPrompt(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device).eval()

    sca_module_norm = SCA_Module(embed_dim=768, num_anchors=args.num_norm_anchors, num_patches=1369).to(device)
    sca_module_abn = SCA_Module(embed_dim=768, num_anchors=args.num_abn_anchors, num_patches=1369).to(device)
    
    if "sca_module_norm" in checkpoint and "sca_module_abn" in checkpoint:
        sca_module_norm.load_state_dict(checkpoint["sca_module_norm"])
        sca_module_abn.load_state_dict(checkpoint["sca_module_abn"])
        logger.info("✅ 成功加载 Dual-SCA 权重！")
    else:
        logger.warning("⚠️ 警告：未在 Checkpoint 中找到 Dual-SCA 权重！")
        
    sca_module_norm.eval()
    sca_module_abn.eval()

    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]['imgs_masks'].append(gt_mask)
        results[cls_name]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
            image_features = image_features.float()
            patch_features = patch_features.float()
            image_features = F.normalize(image_features, dim=-1, eps=1e-8)
            patch_features_norm = F.normalize(patch_features, dim=-1, eps=1e-8)

            prompts_pos, prompts_neg, tokenized_prompt_pos, tokenized_prompt_neg, compound_prompts_text, _ = prompt_learner.forward()
            text_features_pos = model.encode_text_learn(prompts_pos, tokenized_prompt_pos, compound_prompts_text).float()
            text_features_neg = model.encode_text_learn(prompts_neg, tokenized_prompt_neg, compound_prompts_text).float()

            text_features_all = torch.cat([text_features_pos, text_features_neg], dim=0)
            text_features_all = F.normalize(text_features_all, dim=-1, eps=1e-8)

            sim1_all = torch.matmul(patch_features_norm, text_features_all.T) 
            sim1_norm = sim1_all[:, :, 0:1] 
            sim1_abn = sim1_all[:, :, 1:].max(dim=-1, keepdim=True).values 
            
            similarity1 = torch.cat([sim1_norm, sim1_abn], dim=-1) 
            similarity1 = (similarity1 / 0.07).softmax(dim=-1) 
            
            similarity_map1 = AnomalyCLIP_lib.get_similarity_map(similarity1[1:, :], args.image_size).squeeze(0)
            map_max_score1 = similarity1[1:, :, 1].max(dim=0).values

            B = image.shape[0]
            p_kv = patch_features_norm[1:, :, :].permute(1, 0, 2) 
            
            sim_abn_for_mask = torch.matmul(p_kv, text_features_neg.T).max(dim=-1).values 
            valid_norm_mask = sim_abn_for_mask < args.norm_exclusion_thresh
            
            t_query_pos = text_features_pos.unsqueeze(0).expand(B, -1, -1)
            enhanced_pos = sca_module_norm(t_query_pos, p_kv, threshold=None, ext_mask=valid_norm_mask) 
            
            t_query_neg = text_features_neg.unsqueeze(0).expand(B, -1, -1)
            enhanced_neg = sca_module_abn(t_query_neg, p_kv, threshold=args.sca_threshold, fallback_penalty=args.fallback_penalty)

            sim2_norm = (p_kv * enhanced_pos).sum(dim=-1) 
            sim2_abn_all = torch.matmul(p_kv, enhanced_neg.transpose(1, 2)) 
            sim2_abn = sim2_abn_all.max(dim=-1).values 
            
            similarity2 = torch.stack([sim2_norm, sim2_abn], dim=-1) 
            similarity2 = (similarity2 / 0.07).softmax(dim=-1)       
            similarity2 = similarity2.permute(1, 0, 2)               
            
            similarity_map2 = AnomalyCLIP_lib.get_similarity_map(similarity2, args.image_size).squeeze(0) 
            map_max_score2 = similarity2[:, :, 1].max(dim=0).values

            img_sim_all = torch.matmul(image_features, text_features_all.T) 
            img_sim_norm = img_sim_all[:, 0:1]
            img_sim_abn = img_sim_all[:, 1:].max(dim=-1, keepdim=True).values
            img_sim_stacked = torch.cat([img_sim_norm, img_sim_abn], dim=-1) 
            
            text_probs = img_sim_stacked / 0.07
            tmp = text_probs.softmax(-1)[:, 1]

            map_max_score = (2 * map_max_score1 + map_max_score2) / 3.0
            
            # 🟢 动态权重融合
            score2 = args.global_score_weight * tmp + (1.0 - args.global_score_weight) * map_max_score

            anomaly_map = (similarity_map1[1, :] + 1 - similarity_map1[0, :] + similarity_map2[1, :] + 1 - similarity_map2[0, :]) / 4.0
            
            anomaly_map = torch.nan_to_num(anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
            score2 = torch.nan_to_num(score2, nan=0.0, posinf=0.0, neginf=0.0)
            
            am_np = anomaly_map.unsqueeze(0).unsqueeze(0).detach().cpu().numpy()
            anomaly_map_smooth = torch.Tensor(gaussian_filter(am_np, sigma=args.sigma))
            
            results[cls_name]['pr_sp'].append(score2.detach().cpu().item())
            results[cls_name]['anomaly_maps'].append(anomaly_map_smooth)

    table_ls = []
    px_auroc_ls, px_aupro_ls, im_auroc_ls, im_ap_ls = [], [], [], []
    
    for obj in obj_list:
        table = [obj]
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        
        px_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
        px_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
        im_auroc = image_level_metrics(results, obj, "image-auroc")
        im_ap = image_level_metrics(results, obj, "image-ap")

        table.extend([f"{px_auroc*100:.1f}", f"{px_aupro*100:.1f}", f"{im_auroc*100:.1f}", f"{im_ap*100:.1f}"])
        table_ls.append(table)
        
        px_auroc_ls.append(px_auroc)
        px_aupro_ls.append(px_aupro)
        im_auroc_ls.append(im_auroc)
        im_ap_ls.append(im_ap)

    mean_row = ['Mean', 
                f"{np.mean(px_auroc_ls)*100:.1f}", 
                f"{np.mean(px_aupro_ls)*100:.1f}", 
                f"{np.mean(im_auroc_ls)*100:.1f}", 
                f"{np.mean(im_ap_ls)*100:.1f}"]
    table_ls.append(mean_row)

    headers = ['Objects', 'Pixel_AUROC', 'Pixel_AUPRO', 'Image_AUROC', 'Image_AP']
    logger.info("\n%s", tabulate(table_ls, headers=headers, tablefmt="pipe"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("FA-Prompt-DualSCA-Ultimate Test")
    parser.add_argument("--data_path", type=str, default="./data/visa")
    parser.add_argument("--save_path", type=str, default='./results/')
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--sigma", type=int, default=10)
    parser.add_argument("--seed", type=int, default=111)
    
    # 🟢 确保测试时传入与训练模型匹配的参数
    parser.add_argument("--sca_threshold", type=float, default=0.6, help="推理阶段固定的异常截断阈值")
    parser.add_argument("--fallback_penalty", type=float, default=0.1, help="保底正常 Patch 的惩罚权重")
    parser.add_argument("--global_score_weight", type=float, default=0.4, help="Image Level 得分占比权重")
    parser.add_argument("--norm_exclusion_thresh", type=float, default=0.6, help="排雷阻断阈值")
    parser.add_argument("--num_abn_anchors", type=int, default=10, help="异常 SCA 模块的 Anchor 个数")
    parser.add_argument("--num_norm_anchors", type=int, default=1, help="正常 SCA 模块的 Anchor 个数")
    
    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)