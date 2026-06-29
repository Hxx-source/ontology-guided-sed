import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicAttention(nn.Module):
    """
    Dynamic Attention (DA) module from the DAAPNet paper.
    Consists of Channel Attention and Spatial Attention.
    """
    def __init__(self, in_channels, reduction=16):
        super(DynamicAttention, self).__init__()
        
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_weight = self.sigmoid(avg_out + max_out)
        x_ca = x * channel_weight
        
        # Spatial Attention
        avg_spatial = torch.mean(x_ca, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_weight = self.sigmoid(self.spatial_conv(spatial_cat))
        
        out = x_ca * spatial_weight
        return out

class BAConv(nn.Module):
    """
    Bidirectional Asymmetric Convolution (BA-Conv)
    """
    def __init__(self, in_channels, out_channels):
        super(BAConv, self).__init__()
        
        self.time_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.freq_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.center_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        t_feat = self.time_path(x)
        f_feat = self.freq_path(x)
        c_feat = self.center_conv(x)
        
        fused = torch.cat([t_feat, f_feat, c_feat], dim=1)
        out = self.fuse_conv(fused)
        return out

class DAAPNetCNN(nn.Module):
    """
    The full DAAPNet architecture for replacing CNN.
    """
    def __init__(self, in_channels=1):
        super(DAAPNetCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            BAConv(in_channels, 64),
            nn.MaxPool2d(kernel_size=(2, 2)),
            DynamicAttention(64)
        )
        self.layer2 = nn.Sequential(
            BAConv(64, 128),
            nn.MaxPool2d(kernel_size=(2, 2)),
            DynamicAttention(128)
        )
        self.layer3 = nn.Sequential(
            BAConv(128, 256),
            nn.MaxPool2d(kernel_size=(1, 2)),
            DynamicAttention(256)
        )
        self.layer4 = nn.Sequential(
            BAConv(256, 512),
            nn.MaxPool2d(kernel_size=(1, 2)),
            DynamicAttention(512)
        )
        self.out_channels = 512

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Average pooling over the frequency dimension
        # Assuming input shape [Batch, Channels, Time, Freq]
        x = torch.mean(x, dim=3, keepdim=True)
        return x

