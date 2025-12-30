import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader

def resize_image(image, target_size, multiple_of, crop=None, mode=cv2.INTER_CUBIC):
    h, w = image.shape[:2]
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    new_h = new_h if new_h % multiple_of == 0 else (new_h // multiple_of + 1) * multiple_of
    new_w = new_w if new_w % multiple_of == 0 else (new_w // multiple_of + 1) * multiple_of
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=mode)
    if crop is not None:
        # expect crop to be (height, width)
        croped_size = (crop[0], crop[1])
        croped_h = (new_h - croped_size[0]) // 2
        croped_w = (new_w - croped_size[1]) // 2
        resized_image = resized_image[croped_h:croped_h + croped_size[0], croped_w:croped_w + croped_size[1]]
    return resized_image
    
def normalize_image(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (image - mean) / std
    return normalized_image

class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_csv, student_input_size=296, teacher_input_size=518, provide_depth=False):
        # self.data_path = data_path
        # self.images = sorted(glob.glob(os.path.join(data_path, "*.jpg")))
        # self.depths = sorted(glob.glob(os.path.join(data_path, "*.png")))
        # assert len(self.images) == len(self.depths), f"影像數量 {len(self.images)} 與深度數量 {len(self.depths)} 不一致！"
        self.data_path = data_path
        with open(data_csv, 'r') as f:
            lines = f.read().strip().splitlines()
        self.pairs = [line.split(',') for line in lines]
        assert all(len(p) == 2 for p in self.pairs), "CSV 檔案格式錯誤，每行應包含影像與深度圖路徑"
        self.student_input_size = student_input_size
        self.teacher_input_size = teacher_input_size
        self.student_multiple_of = 16
        self.teacher_multiple_of = 14
        self.provide_depth = provide_depth

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, depth_path = self.pairs[idx]
        image_path = os.path.join(self.data_path, image_path)
        depth_path = os.path.join(self.data_path, depth_path)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        image = image / 255.0
        # 判斷 dtype
        if depth.dtype == np.uint8:
            depth = depth.astype(np.float32)# / 255.0
            bit_depth = 8
        elif depth.dtype == np.uint16 or depth.dtype == np.int32:
            depth = depth.astype(np.float32)# / 65535.0
            bit_depth = 16
        else:
            # 其他型態
            raise ValueError(f"Unsupported image dtype: {depth.dtype}")
        # print(f"Depth image bit depth: {bit_depth}")
        # print(f"Depth image min: {depth.min()}, max: {depth.max()}")
        
        assert image.shape[:2] == depth.shape[:2], f"影像與深度圖尺寸不匹配！"
        
        if len(depth.shape) == 2:
            depth = np.expand_dims(depth, axis=-1)
        image_original_shape = (image.shape[0],image.shape[1])
        depth_original_shape = (depth.shape[0],depth.shape[1])

        resize_func = resize_image
        student_image = resize_func(image, self.student_input_size, self.student_multiple_of)
        teacher_image = resize_func(image, self.teacher_input_size, self.teacher_multiple_of)
        
        # # DA if needed

        norm_func = normalize_image
        student_image = norm_func(student_image)
        teacher_image = norm_func(teacher_image)

        student_image = torch.from_numpy(np.ascontiguousarray(np.transpose(student_image, (2, 0, 1))).astype(np.float32))
        teacher_image = torch.from_numpy(np.ascontiguousarray(np.transpose(teacher_image, (2, 0, 1))).astype(np.float32))
        
        if self.provide_depth:
            depth = torch.from_numpy(np.ascontiguousarray(np.transpose(depth, (2, 0, 1))).astype(np.float32))
            return {'student_image': student_image,
                    'teacher_image': teacher_image,
                    'depth': depth,
                    'image_original_shape': image_original_shape,
                    'depth_original_shape': depth_original_shape,}

        return {'student_image': student_image,
                'teacher_image': teacher_image,
                'image_original_shape': image_original_shape,}


if __name__ == '__main__':
    train_dataset = DepthDataset("./dataset", data_csv="./dataset/data/nyu2_train.csv", provide_depth=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)

    # test_dataset = DepthDataset("./dataset", data_csv="./dataset/data/nyu2_test.csv")
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    for i, sample in enumerate(train_loader):
        student_image = sample['student_image']
        teacher_image = sample['teacher_image']
        print(f"Student image batch shape: {student_image.shape}")
        print(f"Teacher image batch shape: {teacher_image.shape}")
        student_image = student_image.to('cuda', non_blocking=True)
        teacher_image = teacher_image.to('cuda', non_blocking=True)
        depth = sample['depth']
        print(f"Depth image min: {depth.min()}, max: {depth.max()}")
        if i == 2:
            break
        # Student image batch shape: torch.Size([8, 3, 304, 400])
        # Teacher image batch shape: torch.Size([8, 3, 518, 700])