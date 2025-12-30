import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2
from torch import nn

teacher_features = []

# 518/14=37
# 2072=37*56
def feature_hook(module, input, output):
    global teacher_features
    # print(input) #(features, patch_h, patch_w)
    # print(type(input[0]), type(input[1]), type(input[2]), len(input)) #tuple, int, int, 3
    # print(type(input[0][0]), len(input[0])) #tuple, 4
    # print(type(input[0][0][0]), type(input[0][0][1]), len(input[0][0])) #torch.Tensor, 2
    # for i in range(len(input[0])):
    #     for j in range(len(input[0][i])):
    #         print(f"input[0][{i}][{j}] shape: {input[0][i][j].shape}")
    # input[0][0][0] shape: torch.Size([1, 2072, 768])
    # input[0][0][1] shape: torch.Size([1, 768])
    # input[0][1][0] shape: torch.Size([1, 2072, 768])
    # input[0][1][1] shape: torch.Size([1, 768])
    # input[0][2][0] shape: torch.Size([1, 2072, 768])
    # input[0][2][1] shape: torch.Size([1, 768])
    # input[0][3][0] shape: torch.Size([1, 2072, 768])
    # input[0][3][1] shape: torch.Size([1, 768])
    
    # features = ((fea, cls), *4)
    # fea shape: torch.Size([1, resized_H/14 * resized_W/14, 768])
    # cls shape: torch.Size([1, 768])
    teacher_feature = [input[0][i][0].clone().detach() for i in range(len(input[0]))]
    for tf in teacher_feature:
        teacher_features.append(tf)
        print(f"成功捕捉到教師特徵張量形狀: {tf.shape}") #torch.Size([1, N, C])

class DepthAnythingV2FeatureExtractor(nn.Module):
    def __init__(self):
        super(DepthAnythingV2FeatureExtractor, self).__init__()
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.encoder = 'vitb'

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        self.depth_anything = DepthAnythingV2(**model_configs[self.encoder])
        self.depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{self.encoder}.pth', map_location='cpu'))
        self.depth_anything = self.depth_anything.to(DEVICE).eval()

        self.teacher_features = []
        self.hook_handle = None
        self.add_hook()
        self.freeze_parameters()
    def freeze_parameters(self):
        for param in self.depth_anything.parameters():
            param.requires_grad = False
    def enc_hook(self, module, input, output):
        teacher_feature = [input[0][i][0].clone().detach() for i in range(len(input[0]))]
        for tf in teacher_feature:
            self.teacher_features.append(tf)
            print(f"成功捕捉到教師特徵張量形狀: {tf.shape}")
    def enc_hook2(self, module, input, output):
        teacher_feature = input[0].clone().detach()
        self.teacher_features.append(teacher_feature)
        # print(f"成功捕捉到教師特徵張量形狀: {teacher_feature.shape}")
    def dec_hook(self, module, input, output):
        teacher_feature = output.clone().detach()
        self.teacher_features.append(teacher_feature)
        # print(f"成功捕捉到教師特徵張量形狀: {teacher_feature.shape}")
    def add_hook(self):
        # self.hook_handle = self.depth_anything.depth_head.register_forward_hook(self.enc_hook)
        self.hook_handle_e1 = self.depth_anything.depth_head.scratch.layer1_rn.register_forward_hook(self.enc_hook2)
        self.hook_handle_e2 = self.depth_anything.depth_head.scratch.layer2_rn.register_forward_hook(self.enc_hook2)
        self.hook_handle_e3 = self.depth_anything.depth_head.scratch.layer3_rn.register_forward_hook(self.enc_hook2)
        self.hook_handle_e4 = self.depth_anything.depth_head.scratch.layer4_rn.register_forward_hook(self.enc_hook2)
        self.hook_handle_d1 = self.depth_anything.depth_head.scratch.refinenet4.register_forward_hook(self.dec_hook)
        self.hook_handle_d2 = self.depth_anything.depth_head.scratch.refinenet3.register_forward_hook(self.dec_hook)
        self.hook_handle_d3 = self.depth_anything.depth_head.scratch.refinenet2.register_forward_hook(self.dec_hook)
        self.hook_handle_d4 = self.depth_anything.depth_head.scratch.refinenet1.register_forward_hook(self.dec_hook)
    def remove_hook(self):
        # self.hook_handle.remove()
        self.hook_handle_e1.remove()
        self.hook_handle_e2.remove()
        self.hook_handle_e3.remove()
        self.hook_handle_e4.remove()
        self.hook_handle_d1.remove()
        self.hook_handle_d2.remove()
        self.hook_handle_d3.remove()
        self.hook_handle_d4.remove()
    def infer_image_(self, raw_image, input_size=518, to_int=False):
        depth = self.depth_anything.infer_image(raw_image, input_size)
        if to_int:
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
        return depth
    def clear_features_(self):
        self.teacher_features = []
    def get_features_(self, clear_after_get=False):
        tfs = self.teacher_features
        if clear_after_get:
            self.clear_features_()
        return tfs
    def extract_features(self, raw_image, input_size=518):
        self.clear_features_()
        if isinstance(raw_image, list):
            with torch.no_grad():
                _ = [self.infer_image_(img, input_size=input_size, to_int=False) for img in raw_image]
            return self.get_features_(clear_after_get=True)
        with torch.no_grad():
            _ = self.infer_image_(raw_image, input_size=input_size, to_int=False)
        return self.get_features_(clear_after_get=True)
    def forward(self, x):
        y = self.depth_anything(x)
        return y, self.get_features_(clear_after_get=True)
    def get_model(self):
        return self.depth_anything
    def count_parameters(self):
        return sum(p.numel() for p in self.depth_anything.parameters() if p.requires_grad)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    dptfe = DepthAnythingV2FeatureExtractor()
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        raw_image = cv2.imread(filename)
        print(f"input shape: {raw_image.shape}")
        feature = dptfe.extract_features(raw_image, args.input_size)
        print(f"extracted {len(feature)} feature maps.")
        for i in range(len(feature)):
            print(f"feature {i} shape: {feature[i].shape}")
    '''
    feature 0 shape: torch.Size([1, 128, 37, 56])
    feature 1 shape: torch.Size([1, 128, 74, 112])
    feature 2 shape: torch.Size([1, 128, 148, 224])
    feature 3 shape: torch.Size([1, 128, 296, 448])
    feature 4 shape: torch.Size([1, 2072, 768])
    feature 5 shape: torch.Size([1, 2072, 768])
    feature 6 shape: torch.Size([1, 2072, 768])
    feature 7 shape: torch.Size([1, 2072, 768])
    '''
    print(f"Total parameters: {dptfe.count_parameters()}")
    print(f"decoder parameters: {sum(p.numel() for p in dptfe.get_model().depth_head.parameters() if p.requires_grad)}")

    '''
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    hook_handle = depth_anything.depth_head.register_forward_hook(feature_hook)
    # hook_handle.remove()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)

        print(f"input shape: {raw_image.shape}") # input shape: (1362, 2048, 3)
        
        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        print(f"output shape: {depth.shape}") # output shape: (1362, 2048)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
    print(teacher_features[-1].shape if len(teacher_features) > 0 else "No features captured yet.")
    # torch.Size([1, 2072, 768])
    '''