import torch
import requests
from PIL import Image
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(
    "Intel/zoedepth-nyu-kitti"
)
model = AutoModelForDepthEstimation.from_pretrained(
    "Intel/zoedepth-nyu-kitti",
    device_map="auto"
)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").to(model.device)

teacher_features = []
def feature_hook(module, input, output):
    global teacher_features
    # print(output) #(tensor, None)
    teacher_feature = output[0].clone().detach()
    teacher_features.append(teacher_feature) # 複製並分離張量以供後續使用
    # print(f"成功捕捉到教師特徵張量形狀: {teacher_feature.shape}") #torch.Size([1, 769, 1024])

# 註冊 Hook
hook_handle = [model.backbone.encoder.layer[i].register_forward_hook(feature_hook) for i in range(24)]

# hook_handle[i].remove()

# neck輸入格式
# self.neck(hidden_states, patch_height, patch_width)
# hidden_states=tuple(hidden_feature[5],hidden_feature[11],hidden_feature[17],hidden_feature[23])
# patch_height=input_height//16 (24)
# patch_width=input_width//16 (32)
# teacher_features_ = []
# def feature_hook_(module, input, output):
#     global teacher_features_
#     # print(input) #((tensor,tensor,tensor,tensor), patch_heights=24, patch_widths=32)
#     teacher_feature_ = [input[0][i].clone().detach() for i in range(len(input[0]))]
#     for i in range(len(input[0])): # len(input[0])=4
#         teacher_features_.append(teacher_feature_[i])
#     for tf in teacher_feature_:
#         print(f"成功捕捉到教師特徵張量形狀: {tf.shape}") #torch.Size([1, 769, 1024])
# hook_handle_ = [model.neck.register_forward_hook(feature_hook_) for i in range(1)]

with torch.no_grad():
  outputs = model(**inputs) #model(pixel_values=inputs.pixel_values)

# interpolate to original size and visualize the prediction
## ZoeDepth dynamically pads the input image, so pass the original image size as argument
## to `post_process_depth_estimation` to remove the padding and resize to original dimensions.
post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    source_sizes=[(image.height, image.width)],
)

predicted_depth = post_processed_output[0]["predicted_depth"]
depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
depth = depth.detach().cpu().numpy() * 255
# Image.fromarray(depth.astype("uint8")).show() #顯示深度圖
# print("depth shape:", depth.shape) # 原始(686, 960) -> 最終(686, 960) 不變

# print(inputs.pixel_values.shape) #torch.Size([1, 3, 384, 512])
# print(outputs.predicted_depth.shape) #torch.Size([1, 384, 512])

# print(model.backbone.encoder)

# for i in range(24):
#     print(teacher_features[i].shape) # torch.Size([1, 769, 1024]) #769 = 1(CLS TOKEN) + 384/16 * 512/16, 1024 = constant

# backbone_cfg = model.config.backbone_config
# # 查看 PE 相關參數
# print("use_absolute_position_embeddings:", backbone_cfg.use_absolute_position_embeddings) # False
# print("use_relative_position_bias:", backbone_cfg.use_relative_position_bias) #True
# BEiT Transformer
# T5-style
# 此種PE是attn計算時的bias，只在計算attn時生效
# 因此考慮不加PE

print(sum(p.numel() for p in model.parameters()))
print(sum(p.numel() for p in model.backbone.parameters()))
print(sum(p.numel() for p in model.backbone.parameters()) / sum(p.numel() for p in model.parameters()))