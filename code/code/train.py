import os 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms 
from PIL import Image 
from model import UNetGenerator, PatchGANDiscriminator 
from security import encrypt_model, decrypt_model, 
embed_watermark 
from torchvision.transforms import ToPILImage 
from Crypto.Random import get_random_bytes 
# ========================= 
# Custom Dataset (Real‐World RGB↔Depth) 
# ========================= 
class TwoDThreeDDataset(Dataset): 
   def __init__(self, root_dir, split="train", 
transform=None): 
       """ 
       root_dir/ 
         train/ 
           2d/   → RGB images (256×256) 
           3d/   → Depth maps (256×256) 
         val/ 
           2d/ 
           3d/ 
       """ 
       self.two_d_dir = os.path.join(root_dir, split, 
"2d") 
       self.three_d_dir = os.path.join(root_dir, split, 
"3d") 
       self.transform = transform 
 
       valid_exts = (".png", ".jpg", ".jpeg") 
       all_files = sorted(os.listdir(self.two_d_dir)) 
       self.filenames = [f for f in all_files if 
f.lower().endswith(valid_exts)] 
 
   def __len__(self): 
 return len(self.filenames) 
 
   def __getitem__(self, idx): 
       filename = self.filenames[idx] 
       img2d_path = os.path.join(self.two_d_dir, 
filename) 
       img3d_path = os.path.join(self.three_d_dir, 
filename) 
 
       image2d = Image.open(img2d_path).convert("RGB") 
       image3d = Image.open(img3d_path).convert("L") 
 
       if self.transform: 
           image2d = self.transform(image2d) 
           image3d = self.transform(image3d) 
 
       return image2d, image3d 
 
 
# ========================= 
# PSNR & SSIM (for validation) 
# ========================= 
def calculate_psnr(pred, target): 
   mse = torch.mean((pred - target) ** 2) 
   if mse == 0: 
       return 100 
   return 20 * torch.log10(1.0 / torch.sqrt(mse)) 
def calculate_ssim(pred, target): 
   from math import exp 
   C1 = 0.01 ** 2 
   C2 = 0.03 ** 2 
 
   mu_x = torch.mean(pred) 
   mu_y = torch.mean(target) 
   sigma_x = torch.var(pred) 
   sigma_y = torch.var(target) 
   sigma_xy = torch.mean((pred - mu_x) * (target - mu_y)) 
 
   ssim_num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + 
C2) 
   ssim_den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + 
sigma_y + C2) 
   return ssim_num / ssim_den 
 
 
if __name__ == "__main__": 
   # ========================= 
   # Training Parameters 
   # ========================= 
   device = torch.device("cuda" if 
torch.cuda.is_available() else "cpu") 
   epochs = 50 
   batch_size = 8            # Adjust based on your 
GPU/CPU memory 
 learning_rate = 2e-4 
   root_data_dir = "./data"  # Make sure data/train/2d, 
data/train/3d, etc. exist 
 
   # ========================= 
   # Model Initialization 
   # ========================= 
   generator = UNetGenerator(in_channels=3, 
out_channels=1).to(device) 
   discriminator = 
PatchGANDiscriminator(in_channels=4).to(device) 
 
   criterion_GAN = nn.BCEWithLogitsLoss() 
   criterion_L1 = nn.L1Loss() 
 
   optimizer_G = optim.Adam(generator.parameters(), 
lr=learning_rate, betas=(0.5, 0.999)) 
   optimizer_D = optim.Adam(discriminator.parameters(), 
lr=learning_rate, betas=(0.5, 0.999)) 
 
   # ========================= 
   # DataLoader Setup (256×256) with num_workers=0 
   # ========================= 
   transform = transforms.Compose([ 
       transforms.Resize((256, 256)), 
       transforms.ToTensor(), 
   ])
 train_dataset = TwoDThreeDDataset(root_data_dir, 
split="train", transform=transform) 
   val_dataset   = TwoDThreeDDataset(root_data_dir, 
split="val",   transform=transform) 
 
   train_loader = DataLoader( 
       train_dataset, 
       batch_size=batch_size, 
       shuffle=True, 
       num_workers=0,       # Avoid worker‐spawn issues 
on macOS 
       pin_memory=False     # MPS doesn’t support 
pin_memory 
   ) 
   val_loader = DataLoader( 
       val_dataset, 
       batch_size=batch_size, 
       shuffle=False, 
       num_workers=0, 
       pin_memory=False 
   ) 
 
   # ========================= 
   # Training & Validation Loop 
   # ========================= 
   best_val_loss = float("inf") 
 
   for epoch in range(epochs): 
generator.train() 
       discriminator.train() 
 
       running_G_loss = 0.0 
       running_D_loss = 0.0 
 
       for i, (imgs2d, imgs3d) in 
enumerate(train_loader): 
           imgs2d = imgs2d.to(device)  # [B, 3, 256, 256] 
           imgs3d = imgs3d.to(device)  # [B, 1, 256, 256] 
 
           # ----------------------------------- 
           #  Train Generator 
           # ----------------------------------- 
           optimizer_G.zero_grad() 
           gen_3d = generator(imgs2d)                # 
[B,1,256,256] 
           fake_pair = torch.cat((imgs2d, gen_3d), 1) # 
[B,4,256,256] 
           pred_fake = discriminator(fake_pair)       # 
[B,1,16,16] for PatchGAN 
 
           valid = torch.ones(pred_fake.shape, 
device=device) 
           loss_G_GAN = criterion_GAN(pred_fake, valid) 
           loss_G_L1  = criterion_L1(gen_3d, imgs3d) 
           loss_G = loss_G_GAN + 100 * loss_G_L1 
           loss_G.backward() 
 optimizer_G.step() 
 
           # ----------------------------------- 
           #  Train Discriminator 
           # ----------------------------------- 
           optimizer_D.zero_grad() 
           real_pair = torch.cat((imgs2d, imgs3d), 1) 
           pred_real = discriminator(real_pair)       # 
[B,1,16,16] 
           valid = torch.ones(pred_real.shape, 
device=device) 
           loss_real = criterion_GAN(pred_real, valid) 
 
           fake_detach = discriminator(torch.cat((imgs2d, 
gen_3d.detach()), 1)) 
           fake_labels = torch.zeros(fake_detach.shape, 
device=device) 
           loss_fake = criterion_GAN(fake_detach, 
fake_labels) 
 
           loss_D = 0.5 * (loss_real + loss_fake) 
           loss_D.backward() 
           optimizer_D.step() 
 
           running_G_loss += loss_G.item() 
           running_D_loss += loss_D.item() 
 
       avg_G_loss = running_G_loss / len(train_loader) 
 avg_D_loss = running_D_loss / len(train_loader) 
 
       # ====================== 
       #  Validation 
       # ====================== 
       generator.eval() 
       val_loss_L1 = 0.0 
       val_psnr = 0.0 
       val_ssim = 0.0 
 
       with torch.no_grad(): 
           for imgs2d, imgs3d in val_loader: 
               imgs2d = imgs2d.to(device) 
               imgs3d = imgs3d.to(device) 
 
               gen_3d = generator(imgs2d) 
               val_loss_L1 += criterion_L1(gen_3d, 
imgs3d).item() 
 
               gen_norm = (gen_3d + 1) / 2 
               gt_norm  = (imgs3d + 1) / 2 
 
               val_psnr += calculate_psnr(gen_norm, 
gt_norm).item() 
               val_ssim += calculate_ssim(gen_norm, 
gt_norm).item() 
 avg_val_L1 = val_loss_L1 / len(val_loader) if 
len(val_loader) > 0 else 0 
       avg_val_psnr = val_psnr / len(val_loader) if 
len(val_loader) > 0 else 0 
       avg_val_ssim = val_ssim / len(val_loader) if 
len(val_loader) > 0 else 0 
 
       print( 
           f"Epoch [{epoch+1}/{epochs}]  " 
           f"G_loss: {avg_G_loss:.4f}  D_loss: 
{avg_D_loss:.4f}  " 
           f"Val_L1: {avg_val_L1:.4f}  Val_PSNR: 
{avg_val_psnr:.4f}  Val_SSIM: {avg_val_ssim:.4f}" 
       ) 
 
       # Save best checkpoints (only if validation set is 
nonempty) 
       if len(val_loader) > 0 and avg_val_L1 < 
best_val_loss: 
           best_val_loss = avg_val_L1 
           os.makedirs("saved_models", exist_ok=True) 
           torch.save(generator.state_dict(), 
"saved_models/generator_best.pth") 
           torch.save(discriminator.state_dict(), 
"saved_models/discriminator_best.pth") 
 
   # ========================= 
   # Encrypt the Best Generator Weights 
 # ========================= 
   aes_key = get_random_bytes(32) 
   os.makedirs("encrypted_models", exist_ok=True) 
   encrypt_model(generator, aes_key, 
"encrypted_models/generator_encrypted.bin") 
   print("Encrypted generator saved to 
encrypted_models/generator_encrypted.bin") 
 
   # ========================= 
   # Optional: Sample Inference + Watermarking 
   # ========================= 
   # If you have a “sample_2d.png” in the root (256×256 
RGB), this will produce output_3d.png 
   # and output_3d_watermarked.png. 
   sample_path = "sample_2d.png" 
   if os.path.isfile(sample_path): 
       generator.eval() 
       sample_img = 
Image.open(sample_path).convert("RGB") 
       sample_tensor = 
transform(sample_img).unsqueeze(0).to(device)  # 
[1,3,256,256] 
       with torch.no_grad(): 
           fake_3d = generator(sample_tensor)  # 
[1,1,256,256] 
 
       to_pil = ToPILImage() 
fake_3d_img = 
to_pil((fake_3d.squeeze(0).cpu().clamp(-1, 1) + 1) / 2) 
fake_3d_img.save("output_3d.png") 
embed_watermark("output_3d.png", "© 
Hunt_My_Vision", "output_3d_watermarked.png") 
print("Saved `output_3d.png` and 
`output_3d_watermarked.png`.")
