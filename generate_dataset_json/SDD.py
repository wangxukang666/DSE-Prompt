import os
import json

class SDDSolver(object):
    # 根据您的数据结构，这里类别名称可以保持不变，但路径处理需要调整
    CLSNAMES = ['electrical commutators']

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = {'train': {}, 'test': {}}
        anomaly_samples = 0
        normal_samples = 0
        
        for cls_name in self.CLSNAMES:
            # 根据您的目录结构，根目录下直接是kos01-kos16，没有'electrical commutators'子目录
            # 因此，将cls_dir直接设为根目录
            cls_dir = self.root
            
            # 获取所有kos文件夹
            species = [d for d in os.listdir(cls_dir) 
                      if os.path.isdir(os.path.join(cls_dir, d)) and d.startswith('kos')]
            species.sort()  # 按数字排序
            
            for phase in ['test']:  # 假设所有数据均为测试集，如需训练集请根据实际划分调整
                cls_info = []
                
                for specie in species:
                    # 构建当前kos文件夹路径
                    specie_path = os.path.join(cls_dir, specie)
                    
                    # 判断是否为异常：这里假设所有kos文件夹均为异常样本
                    # 如有正常样本，请根据实际结构调整条件
                    is_abnormal = True  # 可根据文件夹名或其他条件调整
                    
                    # 获取图像文件列表（假设为.png或.jpg格式）
                    img_names = [f for f in os.listdir(specie_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    img_names.sort()
                    
                    # 假设标注文件位于同一目录，且与图像文件名相对应
                    # 如果标注文件在其他位置，请相应调整路径
                    mask_names = []
                    if is_abnormal:
                        # 这里假设标注文件也在同一文件夹，且以'_mask'或类似后缀区分
                        # 请根据实际标注文件命名规则调整
                        mask_names = [f for f in os.listdir(specie_path) 
                                     if '_mask' in f or '_gt' in f or f.lower().endswith('_annotation.png')]
                        mask_names.sort()
                    
                    for idx, img_name in enumerate(img_names):
                        mask_path = ''
                        if is_abnormal and idx < len(mask_names):
                            # 假设标注文件与图像文件一一对应
                            mask_path = f'{specie}/{mask_names[idx]}'
                        
                        info_img = {
                            'img_path': f'{specie}/{img_name}',
                            'mask_path': mask_path,
                            'cls_name': cls_name,
                            'specie_name': specie,
                            'anomaly': 1 if is_abnormal else 0,
                        }
                        cls_info.append(info_img)
                        
                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples += 1
                            else:
                                normal_samples += 1
                
                info[phase][cls_name] = cls_info
        
        # 保存元数据文件
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + '\n')
        
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)

if __name__ == '__main__':
    # 请确保此路径确实存在且包含kos01-kos16文件夹
    runner = SDDSolver(root='/home/ubuntu/wangxukang/data/KolektorSDD')
    runner.run()