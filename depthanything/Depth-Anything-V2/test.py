import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# 導入您的模型與資料集
from unet import unet
from extract_feature import DepthAnythingV2FeatureExtractor
from dataset import DepthDataset

def denormalize(image_tensor):
    """
    將標準化後的 Tensor 轉回 0~255 的 RGB 圖片 (用於視覺化)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image_tensor.device)
    
    # 反標準化: x * std + mean
    image = image_tensor * std + mean
    image = torch.clamp(image, 0, 1)
    
    # 轉為 Numpy: (C, H, W) -> (H, W, C)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return image

def colorize_depth(depth_map):
    """
    將單通道深度圖上色 (Inferno Colormap)
    """
    # 正規化到 0-255
    min_val = depth_map.min()
    max_val = depth_map.max()
    if max_val - min_val > 1e-5:
        depth_norm = (depth_map - min_val) / (max_val - min_val)
    else:
        depth_norm = np.zeros_like(depth_map)
        
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    # 使用 COLORMAP_INFERNO (類似 Depth Anything 的官方配色)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

def align_to_gt(pred_depth, gt_depth):
    """
    使用最小平方法 (Least Squares) 將預測深度對齊到 GT 的尺度
    pred_depth: numpy array (H, W)
    gt_depth: numpy array (H, W), 數值範圍 0~1 或任意
    """
    # 1. 展平成一維向量，並只取 GT 有效值 (>0) 的部分來計算
    mask = gt_depth > 1e-8
    pred_valid = pred_depth[mask]
    gt_valid = gt_depth[mask]
    
    # 如果沒有有效點，直接返回原值
    if pred_valid.shape[0] < 10:
        return pred_depth

    # 2. 求解線性方程: scale * pred + shift = gt
    # 這是標準的統計學對齊方法
    ones = np.ones_like(pred_valid)
    A = np.vstack([pred_valid, ones]).T # Shape: (N, 2)
    solution = np.linalg.lstsq(A, gt_valid, rcond=None)[0]
    scale, shift = solution[0], solution[1]

    # 3. 將計算出的 scale 和 shift 應用回整張預測圖
    aligned_pred = pred_depth * scale + shift
    
    # 限制在合理範圍 (例如不小於 0)
    aligned_pred = np.maximum(aligned_pred, 0)
    
    return aligned_pred

def test(student_model, teacher_model, test_loader, args):
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 準備資料集
    
    # 2. 載入模型
    print("Loading Models...")
    # Student
    student_model = student_model.to(device)
    if os.path.exists(args.checkpoint):
        student_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded student weights from {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found! Using random weights.")
    
    # Teacher
    teacher_model = teacher_model.to(device)
    
    student_model.eval()
    teacher_model.eval()

    print(f"Start Testing... Output will be saved to {args.output_dir}")
    
    # 3. 推論迴圈
    mae_list = []
    mse_list = []
    rmse_list = []
    mse_teacher_list = []
    mse_student_list = []

    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader)):
            student_image = sample['student_image'].to(device) # (1, 3, 296, 296)
            teacher_image = sample['teacher_image'].to(device) # (1, 3, 518, 518)
            
            # --- Inference ---
            # Teacher Output (作為 Ground Truth / 參考標準)
            teacher_output, _ = teacher_model(teacher_image) 
            
            # Student Output
            student_output = student_model(student_image)
            
            # --- Post-processing ---
            # 將 Student 輸出插值放大到 Teacher 的大小，以便比較
            teacher_output = teacher_output.unsqueeze(1).detach()
            student_output_resized = F.interpolate(
                student_output, 
                size=teacher_output.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )

            depth = sample["depth"].to(device)  # 取得真實深度圖
            # print(f"Depth range in batch: min {depth.min().item():.4f}, max {depth.max().item():.4f}")
            depth_original_shape = sample["depth_original_shape"]
            depth_original_shape = (depth_original_shape[0][0].item(), depth_original_shape[1][0].item())
            teacher_predict = F.interpolate(
                teacher_output, 
                size=depth_original_shape, 
                mode='bilinear', 
                align_corners=False
            )
            student_predict = F.interpolate(
                student_output, 
                size=depth_original_shape, 
                mode='bilinear', 
                align_corners=False
            )
            teacher_np = teacher_predict.cpu().numpy()[0, 0] # 假設 shape 是 [1, 1, H, W] -> [H, W]
            student_np = student_predict.cpu().numpy()[0, 0]
            gt_np = depth.cpu().numpy()[0, 0]

            # 2. 讓預測值去「對齊」GT (解決相對深度的單位問題)
            teacher_aligned = align_to_gt(teacher_np, gt_np)
            student_aligned = align_to_gt(student_np, gt_np)
            # print(f"teacher depth range: min {teacher_aligned.min():.4f}, max {teacher_aligned.max():.4f}")
            # print(f"student depth range: min {student_aligned.min():.4f}, max {student_aligned.max():.4f}")
            # print(f"gt depth range: min {gt_np.min():.4f}, max {gt_np.max():.4f}")
            # --- 計算誤差指標 (可選) ---
            abs_diff = np.abs(student_aligned - teacher_aligned)
            mae = abs_diff.mean().item()
            mse = (abs_diff ** 2).mean().item()
            rmse = np.sqrt((abs_diff ** 2).mean()).item()
            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)
            mse_teacher = np.mean((teacher_aligned - gt_np) ** 2)
            mse_student = np.mean((student_aligned - gt_np) ** 2)
            mse_teacher_list.append(mse_teacher)
            mse_student_list.append(mse_student)

            if i <= args.num_vis - 1:
                # --- 視覺化準備 ---
                # 1. 原始圖片 (取 Teacher 的輸入比較清晰，反標準化)
                rgb_vis = denormalize(teacher_image[0])
                
                # 2. Teacher 深度圖 (轉 numpy)
                tea_depth_vis = teacher_output[0, 0].cpu().numpy()
                tea_depth_vis = colorize_depth(tea_depth_vis)
                
                # 3. Student 深度圖 (轉 numpy)
                stu_depth_vis = student_output_resized[0, 0].cpu().numpy()
                stu_depth_vis = colorize_depth(stu_depth_vis)
                
                # --- 拼圖與存檔 ---
                # 圖片排列: [原圖] [老師預測] [學生預測]
                combined_img = np.hstack((rgb_vis, tea_depth_vis, stu_depth_vis))
                
                # 在圖片上加文字標籤
                h, w, _ = combined_img.shape
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(combined_img, "Input", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(combined_img, "Teacher (Target)", (w//3 + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(combined_img, "Student (Ours)", (2*w//3 + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                save_path = os.path.join(args.output_dir, f"result_{i:04d}.jpg")
                # 存檔 (OpenCV 使用 BGR，所以要轉一下顏色)
                cv2.imwrite(save_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
                if i == args.num_vis - 1:
                    print(f"Saved {args.num_vis} images.")
            
    # --- 整體誤差指標輸出 ---
    print("Testing Completed.")
    print(f"Average MAE: {np.mean(mae_list):.6f}")
    print(f"Average MSE: {np.mean(mse_list):.6f}")
    print(f"Average RMSE: {np.mean(rmse_list):.6f}")
    print(f"Average Teacher MSE: {np.mean(mse_teacher_list):.6f}")
    print(f"Average Student MSE: {np.mean(mse_student_list):.6f}")
    print(f"student performance: {np.mean(mse_teacher_list)/(np.mean(mse_student_list) + 1e-8)*100.0:.4f}% of teacher")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, default='./dataset/data/nyu2_test.csv', help='Path to test CSV')
    parser.add_argument('--checkpoint', type=str, default='ckpt/student_model.pth', help='Path to trained student model')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_vis', type=int, default=10, help='Number of images to visualize')
    args = parser.parse_args()

    test_dataset = DepthDataset("./dataset", data_csv=args.test_csv, provide_depth=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    student_model = unet()
    teacher_model = DepthAnythingV2FeatureExtractor()

    test(student_model, teacher_model, test_loader, args)