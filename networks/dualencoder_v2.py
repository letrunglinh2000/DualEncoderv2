# create the model pytorch 
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead

from .cbam_kan import CBAM
from .convnextv2 import convnextv2_pico

class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out
    

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, mode='bicubic', align_corners=True):
        super(UpConv, self).__init__()
        
        # Convolution layers for local features
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.mode = mode
        self.align_corners = align_corners
        self.cbam = CBAM(out_channels)
        
        # Global context block
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Additional convolution layer for residual connection to match dimensions if necessary
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True, groups=4)
        self.residual_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Interpolation
        x_up = F.interpolate(x, scale_factor=2, mode=self.mode, align_corners=self.align_corners)
        
        # Residual connection
        residual = self.residual_conv(x_up)
        residual = self.residual_bn(residual)

        # Local features
        x_out = self.conv(x_up)
        x_out = self.bn(x_out)
        x_out = self.relu(x_out)

        # Global features
        global_features = self.global_pool(x)
        global_features = self.global_fc(global_features)
        global_features = F.interpolate(global_features, size=x_up.size()[2:], mode=self.mode, align_corners=self.align_corners)

        # Combine local and global features
        x_out = x_out + global_features

        # Apply CBAM
        x_out = self.cbam(x_out)

        # Adding the residual connection
        out = x_out + residual

        return out

class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rates=[1, 6, 12, 18], groups=4):
        super(ASPP, self).__init__()
        
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, dilation=rates[0], groups=groups),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=rates[1], dilation=rates[1], groups=groups),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=rates[2], dilation=rates[2], groups=groups),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=rates[3], dilation=rates[3], groups=groups),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            # nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Conv2d(out_c * 5, out_c, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]
        
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        
        global_avg_pool = self.global_avg_pool(x)
        global_avg_pool = F.interpolate(global_avg_pool, size=size, mode='bicubic', align_corners=True)
        
        x = torch.cat([aspp1, aspp2, aspp3, aspp4, global_avg_pool], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        return x
    
class DualEncoderv2(nn.Module):
    def __init__(self, img_size = 512, in_channels =3, num_classes =1) -> None:
        super().__init__()
        self.num_classes = num_classes 
        self.img_size = img_size
        self.in_channels = in_channels
        self.resnet_encoder = get_encoder(name='resnet34',  # encoder_name
                                        in_channels=3,
                                        depth=5,
                                        weights='imagenet', )
        self.encoder_channels = self.resnet_encoder.out_channels[2:] # (64, 128, 256, 512)
        self.convnext_encoder = convnextv2_pico(pretrained=True)

        self.att33 = AttentionBlock(F_g=self.encoder_channels[2], F_l=self.encoder_channels[2], n_coefficients=self.encoder_channels[2]//2)
        self.att22 = AttentionBlock(F_g=self.encoder_channels[1], F_l=self.encoder_channels[1], n_coefficients=self.encoder_channels[1]//2)
        self.att11 = AttentionBlock(F_g=self.encoder_channels[0], F_l=self.encoder_channels[0], n_coefficients=self.encoder_channels[0]//2)

        self.up4 = UpConv(self.encoder_channels[3], self.encoder_channels[2] )
        self.up3 = UpConv(self.encoder_channels[2], self.encoder_channels[1])
        self.up2 = UpConv(self.encoder_channels[1], self.encoder_channels[0])
        self.up1 = UpConv(self.encoder_channels[0], 32)
        self.aspp = ASPP(32, 32)

        self.final = SegmentationHead(in_channels=32, out_channels=num_classes, upsampling=2)

        self.cbam1 = CBAM(self.encoder_channels[0])
        self.cbam2 = CBAM(self.encoder_channels[1])
        self.cbam3 = CBAM(self.encoder_channels[2])
        self.cbam4 = CBAM(self.encoder_channels[3])
        
    def resnet_features(self, x):
        x = self.resnet_encoder(x)
        features = []
        for i in range(len(x)-2):
            temp = x[i+2]
            features.append(temp)
        return features
    
    def forward(self, x):
        resnet_features = self.resnet_features(x)
        convnext_features = self.convnext_encoder.forward_features(x)
        
        c1 = torch.add(resnet_features[0],convnext_features[0])
        c2 = torch.add(resnet_features[1],convnext_features[1])
        c3 = torch.add(resnet_features[2],convnext_features[2])
        c4 = torch.add(resnet_features[3],convnext_features[3])

        s1 = self.cbam1(c1) 
        s2 = self.cbam2(c2) 
        s3 = self.cbam3(c3) 
        s4 = self.cbam4(c4) 

        up4 = self.up4(s4)

        up33 = self.att33(up4,s3)
        up3 = self.up3(up33)

        up22 = self.att22(up3,s2)
        up2 = self.up2(up22)

        up11 = self.att11(up2,s1)
        up1 = self.up1(up11)
        aspp = self.aspp(up1) 
        final = self.final(aspp)

        return final

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
   

if __name__ == "__main__":
    model = DualEncoderv2()
    model.cuda()
    x = torch.randn((2, 3, 512, 512)).cuda()
    att_final = model(x)
    print("total parameters :",count_parameters(model))
    # from torchsummary import summary
    # summary(model, (3, 512, 512))