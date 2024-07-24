"""
Define my 2D U-Net model structure.
It is formed of an encoder, bottleneck, and decoder.
"""

import torch
import torch.nn as nn

# Unet Conv block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Encoder block, using two conv blocks    
class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncBlock, self).__init__()

        # mid_channels = int(out_channels/2)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
# Unet Upsampling block
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3,
                                         stride=2, padding=1, output_padding=1)

        post_skip = int(in_channels*1.5)
        self.conv1 = ConvBlock(post_skip, out_channels) # after adding skip connection
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        #print('x: ',x.shape)
        #print('skip: ',skip_connection.shape)
        x = torch.cat([x, skip_connection], dim=1)  # Concatenate upsampled with skip connection
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
# Define 2D U-Net architecture
class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels_1):
        super(UNet2D, self).__init__()

        # Encoder
        self.enc1 = EncBlock(in_channels, num_kernels_1)
        self.enc2 = EncBlock(num_kernels_1, num_kernels_1*2)
        self.enc3 = EncBlock(num_kernels_1*2, num_kernels_1*4)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = EncBlock(num_kernels_1*4, num_kernels_1*8)

        # Decoder (upsampling)
        self.dec3 = UpConvBlock(num_kernels_1*8, num_kernels_1*4)
        self.dec2 = UpConvBlock(num_kernels_1*4, num_kernels_1*2)
        self.dec1 = UpConvBlock(num_kernels_1*2, num_kernels_1)

        # Output of Segmentation Head
        self.out_conv = nn.Conv2d(num_kernels_1, out_channels, kernel_size=1)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(num_kernels_1*8, 128)
        self.fc2 = nn.Linear(128, 1)
        
        # Activations
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool = self.pool(enc1)
        enc2 = self.enc2(pool)
        pool = self.pool(enc2)
        enc3 = self.enc3(pool)
        pool = self.pool(enc3)

        # Bottleneck
        bottleneck_out = self.bottleneck(pool)

        # Segmentation Decoder
        seg_out = self.dec3(bottleneck_out, enc3)
        seg_out = self.dec2(seg_out, enc2)
        seg_out = self.dec1(seg_out, enc1)

        # Segmentation Output
        seg_out = self.out_conv(seg_out)
        seg_out = self.sigmoid(seg_out)

        # Classification Output
        class_out = self.global_pool(bottleneck_out)
        class_out = torch.flatten(class_out, 1)
        class_out = self.relu(self.fc1(class_out))
        class_out = self.sigmoid(self.fc2(class_out))

        return seg_out, class_out