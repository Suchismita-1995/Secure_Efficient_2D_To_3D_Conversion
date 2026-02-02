import torch 
import torch.nn as nn 
 
######################## 
# Deeper U-Net Generator (outputs 256×256) 
######################## 
class UNetGenerator(nn.Module): 
   def __init__(self, in_channels=3, out_channels=1, 
base_filters=64): 
       super(UNetGenerator, self).__init__() 
       # Encoder (downsampling) 
       self.enc1 = self._down_block(in_channels, 
base_filters)            # 256→128 
       self.enc2 = self._down_block(base_filters, 
base_filters * 2)       # 128→64 
       self.enc3 = self._down_block(base_filters * 2, 
base_filters * 4)   # 64→32 
       self.enc4 = self._down_block(base_filters * 4, 
base_filters * 8)   # 32→16 
       self.enc5 = self._down_block(base_filters * 8, 
base_filters * 16)  # 16→8 
 
       # Bottleneck 
       self.bottleneck = nn.Sequential(
nn.Conv2d(base_filters * 16, base_filters * 
32, kernel_size=4, stride=2, padding=1, bias=False),  # 
8→4 
           nn.BatchNorm2d(base_filters * 32), 
           nn.ReLU(inplace=True), 
           nn.ConvTranspose2d(base_filters * 32, 
base_filters * 16, kernel_size=4, stride=2, padding=1, 
bias=False),  # 4→8 
           nn.BatchNorm2d(base_filters * 16), 
           nn.ReLU(inplace=True), 
       ) 
 
       # Decoder (upsampling + skip connections) 
       self.dec5 = self._up_block(base_filters * 32, 
base_filters * 8)    # (enc5 + bottleneck) →16 
       self.dec4 = self._up_block(base_filters * 16, 
base_filters * 4)    # (enc4 + dec5) →32 
       self.dec3 = self._up_block(base_filters * 8, 
base_filters * 2)     # (enc3 + dec4) →64 
       self.dec2 = self._up_block(base_filters * 4, 
base_filters)         # (enc2 + dec3) →128 
       self.dec1 = self._up_block(base_filters * 2, 
base_filters // 2)    # (enc1 + dec2) →256 
 
       # Final 1×1 conv: input channels = 
(base_filters//2 + in_channels) 
       self.final_conv = nn.Conv2d(base_filters // 2 + 
in_channels, out_channels, kernel_size=1)
self.tanh = nn.Tanh() 
 
   def _down_block(self, in_ch, out_ch): 
       """ 
       Downsampling block: Conv2d → BatchNorm → LeakyReLU 
       """ 
       return nn.Sequential( 
           nn.Conv2d(in_ch, out_ch, kernel_size=4, 
stride=2, padding=1, bias=False), 
           nn.BatchNorm2d(out_ch), 
           nn.LeakyReLU(0.2, inplace=True) 
       ) 
 
   def _up_block(self, in_ch, out_ch): 
       """ 
       Upsampling block: ConvTranspose2d → BatchNorm → 
ReLU 
       """ 
       return nn.Sequential( 
           nn.ConvTranspose2d(in_ch, out_ch, 
kernel_size=4, stride=2, padding=1, bias=False), 
           nn.BatchNorm2d(out_ch), 
           nn.ReLU(inplace=True) 
       ) 
 
   def forward(self, x): 
       # Encoder 
 e1 = self.enc1(x)   # [B, 64, 128, 128] 
       e2 = self.enc2(e1)  # [B, 128, 64, 64] 
       e3 = self.enc3(e2)  # [B, 256, 32, 32] 
       e4 = self.enc4(e3)  # [B, 512, 16, 16] 
       e5 = self.enc5(e4)  # [B, 1024, 8, 8] 
 
       # Bottleneck 
       bn = self.bottleneck(e5)  # → [B, 1024, 8, 8] 
 
       # Decoder with skip connections 
       d5 = self.dec5(torch.cat([bn, e5], dim=1))  # → 
[B, 512, 16, 16] 
       d4 = self.dec4(torch.cat([d5, e4], dim=1))  # → 
[B, 256, 32, 32] 
       d3 = self.dec3(torch.cat([d4, e3], dim=1))  # → 
[B, 128, 64, 64] 
       d2 = self.dec2(torch.cat([d3, e2], dim=1))  # → 
[B, 64, 128, 128] 
       d1 = self.dec1(torch.cat([d2, e1], dim=1))  # → 
[B, 32, 256, 256] 
 
       # Concatenate with original input (skip 
connection) 
       final_in = torch.cat([d1, x], dim=1)        # → 
[B, 32+3, 256, 256] 
       final = self.final_conv(final_in)           # → 
[B, 1, 256, 256] 
       return self.tanh(final) 
######################## 
# Stronger PatchGAN Discriminator 
######################## 
class PatchGANDiscriminator(nn.Module): 
   def __init__(self, in_channels=4, base_filters=64): 
       super(PatchGANDiscriminator, self).__init__() 
       # A 5‐layer PatchGAN (256×256 → 16×16 patch map) 
       layers = [] 
       layers.append( 
           nn.Sequential( 
               nn.Conv2d(in_channels, base_filters, 
kernel_size=4, stride=2, padding=1), 
               nn.LeakyReLU(0.2, inplace=True) 
           ) 
       )  # 256→128 
       feat = base_filters 
       for i in range(1, 5): 
           out_feat = base_filters * min(2**i, 8) 
           stride = 1 if i == 4 else 2 
           layers.append( 
               nn.Sequential( 
                   nn.Conv2d(feat, out_feat, 
kernel_size=4, stride=stride, padding=1, bias=False), 
                   nn.BatchNorm2d(out_feat), 
                   nn.LeakyReLU(0.2, inplace=True) 
               ) 
 ) 
           feat = out_feat 
       # Final conv (no batchnorm) 
       layers.append(nn.Conv2d(feat, 1, kernel_size=4, 
stride=1, padding=1)) 
       self.model = nn.Sequential(*layers) 
 
   def forward(self, x): 
       return self.model(x)
