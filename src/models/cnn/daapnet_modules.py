import torch
import torch.nn as nn
import torch.nn.functional as F

class BAConv(nn.Module):
    """
    Bidirectional Asymmetric Convolution (BA-Conv) from the paper.
    It separates time-domain and frequency-domain feature extraction using asymmetric kernels.
    """
    def __init__(self, in_channels, out_channels):
        super(BAConv, self).__init__()
        # Time-domain path (1x3 followed by 3x1 or vice versa depending on implementation, here we use 1x3 then 3x1 for one, and 3x1 then 1x3 for another)
        self.time_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Frequency-domain path
        self.freq_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Center path
        self.center_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 conv to fuse the paths
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
    A CNN extractor replacing the traditional CNN using DAAPNet's BA-Conv principles.
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(DAAPNetCNN, self).__init__()
        
        # Example CNN architecture using BAConv instead of standard Conv2d
        self.layer1 = nn.Sequential(
            BAConv(in_channels, 64),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer2 = nn.Sequential(
            BAConv(64, 128),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer3 = nn.Sequential(
            BAConv(128, 256),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.layer4 = nn.Sequential(
            BAConv(256, 512),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        
        self.out_channels = 512

    def forward(self, x):
        # x shape: (Batch, Channel, Time, Freq)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Aggregate frequency dimension (average pooling)
        # Assuming shape after layers is (B, C, T, F), we want (B, C, T, 1)
        x = torch.mean(x, dim=3, keepdim=True)
        return x

