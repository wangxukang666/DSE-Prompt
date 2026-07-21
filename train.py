import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
from FAPrompt import FAPrompt
from loss import FocalLoss, BinaryDiceLoss, BinaryFocalLoss
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
import math

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
                    
                    # 🟢 执行最严厉的软截断惩罚 (如 0.1)
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
        attn = attn * penalty_mask # 🟢 惩罚生效：正常图片仅吸收 10%
        
        U = torch.matmul(attn, patch_features) 
        gate = torch.sigmoid(self.gate_w(text_features)) 
        delta_t = torch.matmul(gate, U) 
        enhanced_text_features = text_features + self.res_scale * delta_t
        
        return F.normalize(enhanced_text_features, dim=-1, eps=1e-8)

def train(args):
    logger = get_logger(args.save_path)
    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    prompt_learner = FAPrompt(model.to("cpu"), AnomalyCLIP_parameters).to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    # 🟢 动态解析 Anchor 数量
    sca_module_norm = SCA_Module(embed_dim=768, num_anchors=args.num_norm_anchors, num_patches=1369).to(device)
    sca_module_abn = SCA_Module(embed_dim=768, num_anchors=args.num_abn_anchors, num_patches=1369).to(device)
    
    optimizer = torch.optim.Adam([
        {'params': prompt_learner.parameters(), 'weight_decay': 1e-4},
        {'params': sca_module_norm.parameters(), 'lr': args.learning_rate * 5.0, 'weight_decay': 1e-4},
        {'params': sca_module_abn.parameters(), 'lr': args.learning_rate * 5.0, 'weight_decay': 1e-4}
    ], lr=args.learning_rate, betas=(0.5, 0.999))

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_fun = BinaryFocalLoss()

    for epoch in tqdm(range(args.epoch)):
        prompt_learner.train()
        sca_module_norm.train()
        sca_module_abn.train()
        
        target_reg_weight = 0.05
        current_reg_weight = target_reg_weight * min(1.0, (epoch + 1) / args.warmup_epochs)
        
        # 🟢 新增：连续动态计算当前 Epoch 的 SCA 阈值 (如前5轮从 0.4 平滑增至 0.7)
        if epoch < args.thresh_warmup_epochs:
            # 当前进度比例 (例如 5轮就是 0, 0.25, 0.5, 0.75, 1.0)
            progress = epoch / max(1, (args.thresh_warmup_epochs - 1)) 
            current_sca_threshold = args.thresh_start + progress * (args.thresh_end - args.thresh_start)
        else:
            # 超过过渡轮数后，一直保持在最高阈值
            current_sca_threshold = args.thresh_end
            
        loss_list, image_loss_list, image_loss_list2, prompt_reg_loss_list = [], [], [], []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1; gt[gt <= 0.5] = 0

            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer = 20)
                image_features = F.normalize(image_features.float(), dim=-1, eps=1e-8)
                patch_features_norm = F.normalize(patch_features.float(), dim=-1, eps=1e-8)

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
            
            similarity_map1 = AnomalyCLIP_lib.get_similarity_map(similarity1[1:, :], args.image_size)
            map_max_score1 = similarity1[1:, :, 1].max(dim=0).values

            B = image.shape[0]
            p_kv = patch_features_norm[1:, :, :].permute(1, 0, 2) 
            
            sim_abn_for_mask = torch.matmul(p_kv, text_features_neg.T).max(dim=-1).values 
            valid_norm_mask = sim_abn_for_mask < args.norm_exclusion_thresh 
            
            t_query_pos = text_features_pos.unsqueeze(0).expand(B, -1, -1)
            enhanced_pos = sca_module_norm(t_query_pos, p_kv, threshold=None, ext_mask=valid_norm_mask) 
            
            t_query_neg = text_features_neg.unsqueeze(0).expand(B, -1, -1)
            # 🟢 传入动态阈值与惩罚系数
            enhanced_neg = sca_module_abn(t_query_neg, p_kv, threshold=current_sca_threshold, fallback_penalty=args.fallback_penalty)

            total_prompt_reg_val = 0
            num_abn = enhanced_neg.size(1) 
            
            if num_abn > 1:
                sim_matrix = F.cosine_similarity(enhanced_neg.unsqueeze(2), enhanced_neg.unsqueeze(1), dim=-1) 
                mask = torch.triu(torch.ones(num_abn, num_abn, device=device), diagonal=1).bool()
                sim_abn_abn = sim_matrix[:, mask] 
                
                loss_div_upper = torch.mean(F.relu(sim_abn_abn - args.div_sim_upper)) 
                loss_div_lower = torch.mean(F.relu(args.div_sim_lower - sim_abn_abn)) 
                loss_div = loss_div_upper + loss_div_lower
                
                total_prompt_reg_val += loss_div

            total_prompt_reg = current_reg_weight * total_prompt_reg_val
            prompt_reg_loss_list.append(total_prompt_reg.item() if isinstance(total_prompt_reg, torch.Tensor) else 0)

            sim2_norm = (p_kv * enhanced_pos).sum(dim=-1) 
            sim2_abn_all = torch.matmul(p_kv, enhanced_neg.transpose(1, 2)) 
            sim2_abn = sim2_abn_all.max(dim=-1).values 
            
            similarity2 = torch.stack([sim2_norm, sim2_abn], dim=-1) 
            similarity2 = (similarity2 / 0.07).softmax(dim=-1)       
            similarity2 = similarity2.permute(1, 0, 2)               
            
            similarity_map2 = AnomalyCLIP_lib.get_similarity_map(similarity2, args.image_size)
            map_max_score2 = similarity2[:, :, 1].max(dim=0).values

            img_sim_all = torch.matmul(image_features, text_features_all.T) 
            img_sim_norm = img_sim_all[:, 0:1]
            img_sim_abn = img_sim_all[:, 1:].max(dim=-1, keepdim=True).values
            img_sim_stacked = torch.cat([img_sim_norm, img_sim_abn], dim=-1) 
            
            text_probs = img_sim_stacked / 0.07 
            image_loss = F.cross_entropy(text_probs, label.long().to(device))
            image_loss_list.append(image_loss.item())

            tmp = text_probs.softmax(-1)[:, 1]
            map_max_score = (2 * map_max_score1 + map_max_score2) / 3
            
            # 🟢 动态权重融合
            score2 = args.global_score_weight * tmp + (1.0 - args.global_score_weight) * map_max_score

            similarity_map1 = torch.clamp(similarity_map1, min=1e-5, max=1.0 - 1e-5)
            similarity_map2 = torch.clamp(similarity_map2, min=1e-5, max=1.0 - 1e-5)
            score2 = torch.clamp(score2, min=1e-5, max=1.0 - 1e-5)

            image_loss2 = loss_fun(score2, label.float().to(device))
            image_loss_list2.append(image_loss2.item())

            loss = 0
            loss += loss_focal(similarity_map1, gt) + loss_dice(similarity_map1[:, 1, :, :], gt) + loss_dice(similarity_map1[:, 0, :, :], 1-gt)
            loss += 0.5 * loss_focal(similarity_map2, gt) + 0.5 * loss_dice(similarity_map2[:, 1, :, :], gt) + 0.5 * loss_dice(similarity_map2[:, 0, :, :], 1 - gt)

            optimizer.zero_grad()
            (loss + image_loss + image_loss2 + total_prompt_reg).backward()
            
            torch.nn.utils.clip_grad_norm_(prompt_learner.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(sca_module_norm.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(sca_module_abn.parameters(), max_norm=1.0)

            optimizer.step()
            loss_list.append(loss.item())

        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, img_loss:{:.4f}, img_loss2:{:.4f}, reg_loss:{:.4f}, thresh:{:.2f}'.format(
                epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list), np.mean(image_loss_list2), np.mean(prompt_reg_loss_list), current_sca_threshold))

        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({
                "prompt_learner": prompt_learner.state_dict(),
                "sca_module_norm": sca_module_norm.state_dict(),
                "sca_module_abn": sca_module_abn.state_dict()
            }, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("FA-Prompt-DualSCA-Ultimate", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa")
    parser.add_argument("--save_path", type=str, default='./checkpoint')
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=10) 
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=0.0001) 
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=111)
    
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--div_sim_lower", type=float, default=0.0)
    parser.add_argument("--div_sim_upper", type=float, default=0.2)
    
    # 🟢 核心动态解析参数
    # 🟢 核心动态解析参数 (已修改为连续动态阈值)
    parser.add_argument("--thresh_start", type=float, default=0.4, help="初始异常阈值 (第1轮)")
    parser.add_argument("--thresh_end", type=float, default=0.7, help="最终异常阈值")
    parser.add_argument("--thresh_warmup_epochs", type=int, default=5, help="阈值动态递增的过渡轮数")
    parser.add_argument("--fallback_penalty", type=float, default=0.1, help="正常图片被迫吸收时的极低惩罚权重 (越低越不污染)")
    #改动
    parser.add_argument("--global_score_weight", type=float, default=0.5, help="Image Level tmp 得分占比权重")
    parser.add_argument("--norm_exclusion_thresh", type=float, default=0.6, help="正常 SCA 排雷阻断阈值")
    parser.add_argument("--num_abn_anchors", type=int, default=10, help="异常 SCA 模块的 Anchor 个数")
    #改动
    parser.add_argument("--num_norm_anchors", type=int, default=1, help="正常 SCA 模块的 Anchor 个数")
    
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)