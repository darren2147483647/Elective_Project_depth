import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import cv2
from torch.utils.data import DataLoader

class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_csv, preprocess=True):
        self.data_path = data_path
        with open(data_csv, 'r') as f:
            lines = f.read().strip().splitlines()
        self.pairs = [line.split(',') for line in lines]
        assert all(len(p) == 2 for p in self.pairs), "CSV 檔案格式錯誤，每行應包含影像與深度圖路徑"
        self.preprocess = preprocess
        self.imgsize = 592
        self.multiple = 16
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, depth_path = self.pairs[idx]
        image_path = os.path.join(self.data_path, image_path)
        depth_path = os.path.join(self.data_path, depth_path)
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        if self.preprocess:
            scale = self.imgsize / min(h,w)
            new_h, new_w = int(h * scale), int(w * scale)
            new_h = new_h if new_h % self.multiple == 0 else (new_h // self.multiple + 1) * self.multiple
            new_w = new_w if new_w % self.multiple == 0 else (new_w // self.multiple + 1) * self.multiple
            x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            x = cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            x = torch.from_numpy(x.transpose((2,0,1))).float()
        return {'image': x, 'origin_h': h, 'origin_w': w}
if __name__ == '__main__':
    train_dataset = DepthDataset("./dataset", data_csv="./dataset/data/nyu2_train.csv")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    