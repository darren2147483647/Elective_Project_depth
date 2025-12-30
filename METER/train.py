import torch
import torch.nn.functional as F
import numpy as np
from architecture import *
from augmentation import *
from loss import *
import os
import glob
from PIL import Image
from tqdm import tqdm

def train(model, device, train_loader, optimizer, compute_loss, epoch, scheduler=None):
    model.to(device)
    for i in range(epoch):# 可在背景時disable進度條
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {i+1}/{epoch}", disable=False)
        model.train()
        for batch_idx, sample in enumerate(loop):
            image, depth = sample['image'], sample['depth']
            image, depth = image.to(device, non_blocking=True), depth.to(device, non_blocking=True)
            pred = model(image)
            pred_up = F.interpolate(pred, size=depth.shape[-2:], mode='bilinear', align_corners=False)
            assert pred_up.shape == depth.shape, f"預測深度圖與真實深度圖尺寸不匹配！{pred_up.shape} vs {depth.shape}"
            loss_depth, loss_ssim, loss_normal, loss_grad = compute_loss(pred_up, depth)
            loss = loss_depth + loss_ssim + loss_normal + loss_grad
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix({
                "loss": loss.item(),
                "-depth": loss_depth.item(),
                "-ssim": loss_ssim.item(),
                "-normal": loss_normal.item(),
                "-grad": loss_grad.item()
            })
        scheduler.step() if scheduler is not None else None
    torch.save(model.state_dict(), 'meter_model2.pth')

import argparse
from torch.utils.data import DataLoader

class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_csv):
        # self.data_path = data_path
        # self.images = sorted(glob.glob(os.path.join(data_path, "*.jpg")))
        # self.depths = sorted(glob.glob(os.path.join(data_path, "*.png")))
        # assert len(self.images) == len(self.depths), f"影像數量 {len(self.images)} 與深度數量 {len(self.depths)} 不一致！"
        self.data_path = data_path
        with open(data_csv, 'r') as f:
            lines = f.read().strip().splitlines()
        self.pairs = [line.split(',') for line in lines]
        assert all(len(p) == 2 for p in self.pairs), "CSV 檔案格式錯誤，每行應包含影像與深度圖路徑"
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, depth_path = self.pairs[idx]
        image_path = os.path.join(self.data_path, image_path)
        depth_path = os.path.join(self.data_path, depth_path)
        image = np.array(Image.open(image_path).convert("RGB"))
        # depth = np.array(Image.open(depth_path).convert("L"))

        depth = Image.open(depth_path)
        depth = np.array(depth)
        # 判斷 dtype
        if depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0
            bit_depth = 8
        elif depth.dtype == np.uint16 or depth.dtype == np.int32:
            depth = depth.astype(np.float32) / 65535.0
            bit_depth = 16
        else:
            # 其他型態可根據需求擴充
            raise ValueError(f"Unsupported image dtype: {depth.dtype}")
        # (480, 640, 3) (480, 640)
        assert image.shape[:2] == depth.shape[:2], f"影像與深度圖尺寸不匹配！"
        
        # print(image.shape, depth.shape)
        # exit()
        if len(depth.shape) == 2:
            depth = np.expand_dims(depth, axis=-1)
        # image_shape = (image.shape[0],image.shape[1])
        # depth_shape = (depth.shape[0],depth.shape[1])
        image_shape = (192, 256)
        depth_shape = (192, 256)
        # augmentation2D的random flip好像有問題 暫不修改
        image, depth = augmentation2D(image, depth, print_info_aug=False)
        image, depth = torch.from_numpy(image).float(), torch.from_numpy(depth).float()
        image = F.interpolate(image.permute(2, 0, 1).unsqueeze(0), size=image_shape, mode='bilinear', align_corners=False).squeeze(0)
        depth = F.interpolate(depth.permute(2, 0, 1).unsqueeze(0), size=depth_shape, mode='nearest').squeeze(0)
        
        # 標準化影像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image / 255.0 - mean) / std
        return {'image': image, 'depth': depth}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train METER model')
    parser.add_argument('--data-path', type=str, default="./dataset", help='Path to the training data')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    args = parser.parse_args()

    train_dataset = DepthDataset(args.data_path, data_csv="./dataset/data/nyu2_train.csv")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = build_METER_model(args.device,"xxs")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    compute_loss = balanced_loss_function(device=args.device)

    train(model, args.device, train_loader, optimizer, compute_loss, args.epochs, scheduler)