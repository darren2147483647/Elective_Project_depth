import torch
import torch.nn as nn
import cv2
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity
import torch.nn.functional as F

# from torchvision.models import list_models
# all_models = list_models()

def calculate_gflops(model, input_tensor):
    with profile(activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True) as prof:
        model(input_tensor)
    total_flops = 0
    for item in prof.key_averages():
        if hasattr(item, "flops"):
            total_flops += item.flops
    return total_flops / 1e9
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

class GlobalExtractor(nn.Module):
    def __init__(self):
        super(GlobalExtractor, self).__init__()
        self.mobilenet = models.mobilenet_v3_small(weights='DEFAULT')
    def forward(self, x):
        return self.mobilenet.features(x)
    
class AddGlobalFeature(nn.Module):
    def __init__(self, in_channel, global_channel = 576, out_channel = None):
        super(AddGlobalFeature,self).__init__()
        if out_channel is None:
            out_channel = in_channel
        self.channel_align = nn.Conv2d(in_channel + global_channel, out_channel, kernel_size=1)
    def forward(self, x, global_feature):
        _, _, stu_h, stu_w = x.shape
        # assert global_feature.shape[1] == 576, "Global feature channel size must be 576"
        global_feature = F.interpolate(global_feature, size=(stu_h, stu_w), mode='bilinear', align_corners=False)
        x = torch.cat((x, global_feature), dim=1)
        x = self.channel_align(x)
        return x

class DsConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DsConv,self).__init__()
        self.dwconv=nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=(3,3),stride=(1,1),padding=(1,1),groups=in_channel,bias=False)
        self.pwconv=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=True)
        self.bn=nn.BatchNorm2d(out_channel)
        self.act=nn.ReLU()
    def forward(self,x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class twoConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(twoConv,self).__init__()
        self.conv=nn.Sequential(
            DsConv(in_channel=in_channel,out_channel=out_channel),
            DsConv(in_channel=out_channel,out_channel=out_channel),
        )
    def forward(self,x):
        return self.conv(x)

class maxPool(nn.Module):
    def __init__(self):
        super(maxPool,self).__init__()
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0))
    def forward(self,x):
        return self.pool(x)

class convTran(nn.Module):
    def __init__(self,in_channel):
        super(convTran,self).__init__()
        self.tran=nn.ConvTranspose2d(in_channels=in_channel,out_channels=in_channel//2,kernel_size=(2,2),stride=(2,2))
    def forward(self,x):
        return self.tran(x)
    
class unet(nn.Module):
    def __init__(self,in_channel=3,out_channel=1):
        super(unet,self).__init__()
        self.hidden_channels=[32,64,128,256]
        self.left=nn.ModuleList()
        self.addfea=nn.ModuleList()
        for hidden_channel in self.hidden_channels:
            self.left.append(twoConv(in_channel,hidden_channel))
            self.addfea.append(AddGlobalFeature(hidden_channel))
            in_channel=hidden_channel
        self.pool=maxPool()
        self.bottle_neck=twoConv(in_channel=in_channel,out_channel=in_channel*2)
        self.bottle_neck_addfea = AddGlobalFeature(in_channel*2)
        in_channel*=2
        self.right=nn.ModuleList()
        for hidden_channel in reversed(self.hidden_channels):
            self.right.append(convTran(in_channel=in_channel))
            self.right.append(twoConv(in_channel,hidden_channel))
            in_channel=hidden_channel
        self.end=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=True)
        self.features=[]

        self.mobilenet = GlobalExtractor()
    def forward(self,x):#nx3x256x256
        save=[]
        self.features=[]
        global_feature = self.mobilenet(x)  # Extract global features using MobileNetV3
        for conv, conv_addfea in zip(self.left, self.addfea):
            x=conv(x)#nx64x256x256 nx128x128x128 nx256x64x64 nx512x32x32
            x_=conv_addfea(x, global_feature) # add global feature on bottleneck
            save.append(x_)
            self.features.append(x_)
            x=self.pool(x)#nx64x128x128 nx128x64x64 nx256x32x32 nx512x16x16
        save.reverse()
        x=self.bottle_neck(x)#nx1024x16x16
        x = self.bottle_neck_addfea(x, global_feature)  # add global feature on bottleneck
        for i in range(len(self.right)):
            if i%2:
                x=self.right[i](x)#nx512x32x32 nx256x64x64 nx128x128x128 nx64x256x256
                self.features.append(x)
            else:
                x=self.right[i](x)#nx512x32x32 nx256x64x64 nx128x128x128 nx64x256x256
                x=torch.concatenate((x,save[i//2]),dim=1)#nx(512+512)x32x32 nx(256+256)x64x64 nx(128+128)x128x128 nx(64+64)x256x256
        x=self.end(x)#nx1x256x256
        return x
    def preprocess(self,device,image,imgsize=592,multiple=16):
        '''
        preprocess from a raw image to model input
        '''
        h,w=image.shape[:2]
        scale=imgsize/min(h,w)
        new_h,new_w=int(h*scale),int(w*scale)
        new_h=new_h if new_h%multiple==0 else (new_h//multiple+1)*multiple
        new_w=new_w if new_w%multiple==0 else (new_w//multiple+1)*multiple
        x=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        x=cv2.resize(x,(new_w,new_h),interpolation=cv2.INTER_CUBIC)
        x=(x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        x=torch.from_numpy(x.transpose((2,0,1))).unsqueeze(0).float()
        x=x.to(device)
        return x,(h,w)


if __name__ == "__main__":
    model=unet(in_channel=3,out_channel=1)
    x=torch.randn((1,3,592,592))
    assert x.shape[-1]%16==0 and x.shape[-2]%16==0, "輸入尺寸需為16的倍數"
    y=model(x)
    print(y.shape)  #torch.Size([1, 1, 256, 256])
    # for feature in model.features:
    #     print(feature.shape)
    '''
    torch.Size([1, 64, 256, 256])
    torch.Size([1, 128, 128, 128])
    torch.Size([1, 256, 64, 64])
    torch.Size([1, 512, 32, 32])
    torch.Size([1, 512, 32, 32])
    torch.Size([1, 256, 64, 64])
    torch.Size([1, 128, 128, 128])
    torch.Size([1, 64, 256, 256])
    '''
    para = count_parameters(model)
    print(f"Total parameters: {para} M") # 通道減半後4.980804 M # 11.0785 M > 10M
    gflop = calculate_gflops(model, x) # 通道減半後 8.36867916 # 23.033145592 > 10 G
    print(f"GFLOPs: {gflop}")

    # mobilenet = GlobalExtractor()
    # x_mobilenet = torch.randn(1, 3, 224, 224)
    # params_mobilenet = count_parameters(mobilenet)
    # print(f"MobileNet Parameters: {params_mobilenet:.2f} M") # 預期2.5M
    # gflop_mobilenet = calculate_gflops(mobilenet, x_mobilenet)
    # print(f"MobileNet GFLOPs: {gflop_mobilenet:.2f} GFLOPs") # 0.11G
    # y_mobilenet = mobilenet(x_mobilenet)
    # print(f"MobileNet Output Shape: {y_mobilenet.shape}") # torch.Size([1, 576, 7, 7])