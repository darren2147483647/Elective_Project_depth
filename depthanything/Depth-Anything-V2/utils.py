import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAligner(nn.Module):
    def __init__(self, channel_config=None):
        """
        參數:
            channel_config: 一個列表，包含每一層的通道對應 (student_channel, teacher_channel)
            例如: [(32, 96), (64, 192), ...]
        """
        super().__init__()
        if channel_config is None:
            # 預設的通道配置 (根據學生與教師模型的結構)
            channel_config = [
                (32, 96),
                (64, 192),
                (128, 384),
                (256, 768),
                (256, 128),
                (128, 128),
                (64, 128),
                (32, 128),
            ]
        # 使用 ModuleList 來儲存每一層的 1x1 Conv 對齊器
        # 這確保了這些 Conv 層的參數會被 PyTorch 註冊，並能被 Optimizer 更新
        self.aligners = nn.ModuleList([
            nn.Conv2d(stu_c, tea_c, kernel_size=1)
            for stu_c, tea_c in channel_config
        ])

    def forward(self, student_features, teacher_features, loss_fn=F.mse_loss):
        """
        參數:
            student_features: 學生模型特徵列表 [f1, f2, ...]
            teacher_features: 教師模型特徵列表 [f1, f2, ...]
            loss_fn: 損失函數 (預設 MSE)
        回傳:
            total_loss: 加總後的蒸餾損失
        """
        total_loss = 0.0
        
        # 確保特徵層數與設定檔一致
        assert len(student_features) == len(teacher_features) == len(self.aligners), \
            f"特徵層數不匹配: S={len(student_features)}, T={len(teacher_features)}, Aligners={len(self.aligners)}"

        for i, (stu_feat, tea_feat) in enumerate(zip(student_features, teacher_features)):
            # 1. 通道對齊 (Channel Alignment using 1x1 Conv)
            # 形狀變換: [B, S_C, H, W] -> [B, T_C, H, W]
            stu_aligned = self.aligners[i](stu_feat)
            
            # 2. 空間對齊 (Spatial Alignment using Interpolate)
            # 取得教師特徵的高寬 (H_t, W_t)
            target_size = tea_feat.shape[-2:]
            
            # 形狀變換: [B, T_C, S_H, S_W] -> [B, T_C, T_H, T_W]
            stu_resized = F.interpolate(
                stu_aligned, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            # 3. 計算損失
            layer_loss = loss_fn(stu_resized, tea_feat)
            total_loss += layer_loss
            
        return total_loss