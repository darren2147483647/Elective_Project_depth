import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import argparse

from unet import unet
from extract_feature import DepthAnythingV2FeatureExtractor
from dataset import DepthDataset


def train(student_model, teacher_model, train_loader, optimizer, scheduler, compute_loss, args):
    device = args.device
    student_model.to(device)
    teacher_model.to(device)
    student_model.train()
    teacher_model.eval()
    for epoch in range(args.epochs):
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}", disable=False)
        for batch_idx, sample in enumerate(loop):
            student_image, teacher_image = sample['student_image'], sample['teacher_image']
            student_image, teacher_image = student_image.to(device, non_blocking=True), teacher_image.to(device, non_blocking=True)
            with torch.no_grad():
                teacher_output, teacher_features = teacher_model(teacher_image)
            student_output = student_model(student_image)
            student_features = student_model.features
            teacher_output = teacher_output.unsqueeze(1).detach()
            loss = 0.0
            # 將 student_output (例如 256x256) 插值到 teacher_output (例如 518x518)
            student_output_resized = F.interpolate(student_output, size=teacher_output.shape[-2:], mode='bilinear', align_corners=False)
            loss_depth = compute_loss(student_output_resized, teacher_output)
            loss_features = student_model.align_features(student_features, teacher_features)
            loss += loss_depth + args.hyper_para * loss_features
            # for sf, tf in zip(student_features, teacher_features):
            #     print(f"Student feature shape: {sf.shape}, Teacher feature shape: {tf.shape}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix({
                "loss": loss.item(),
            })
        scheduler.step() if scheduler is not None else None
        if (epoch + 1) % 10 == 0:
            torch.save(student_model.state_dict(), f'ckpt/student_model_epoch_{epoch+1}.pth')
    torch.save(student_model.state_dict(), 'ckpt/student_model.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train METER model')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--hyper_para', type=float, default=1.0, help='Hyperparameter for feature alignment loss')
    parser.add_argument('--train_csv', type=str, default='./dataset/data/nyu2_train.csv', help='Path to training CSV')

    args = parser.parse_args()

    train_dataset = DepthDataset("./dataset", data_csv=args.train_csv)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    student_model = unet().to(args.device)
    teacher_model = DepthAnythingV2FeatureExtractor().to(args.device)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    compute_loss = F.mse_loss

    train(student_model, teacher_model, train_loader, optimizer, scheduler, compute_loss, args)

    # student_x, teacher_x = torch.randn((1,3,296,296)), torch.randn((1,3,518,518))
    # student_y = student_model(student_x)
    # student_features = student_model.features
    # teacher_y = teacher_model.depth_anything(teacher_x.to(args.device))
    # teacher_features = teacher_model.get_features_(clear_after_get=True)