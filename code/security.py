import os 
from Crypto.Cipher import AES 
from Crypto.Random import get_random_bytes 
import torch 
import io 
from PIL import Image, ImageDraw, ImageFont 
 
######################## 
# AES Encryption / Decryption 
######################## 
def encrypt_model(model: torch.nn.Module, key: bytes, 
output_path: str): 
""" 
   Encrypts the state_dict of a PyTorch model using AES 
and saves to output_path. 
   """ 
   buffer = io.BytesIO() 
   torch.save(model.state_dict(), buffer) 
   model_bytes = buffer.getvalue() 
 
   iv = get_random_bytes(16) 
   cipher = AES.new(key, AES.MODE_CBC, iv) 
   pad_len = 16 - (len(model_bytes) % 16) 
   model_bytes_padded = model_bytes + bytes([pad_len] * 
pad_len) 
 
   encrypted = cipher.encrypt(model_bytes_padded) 
   # Ensure the directory exists 
   os.makedirs(os.path.dirname(output_path), 
exist_ok=True) 
   with open(output_path, "wb") as f_out: 
       f_out.write(iv + encrypted) 
 
 
def decrypt_model(model: torch.nn.Module, key: bytes, 
encrypted_path: str): 
   """ 
   Decrypts the content at encrypted_path and loads into 
the provided PyTorch model. 
   """
 with open(encrypted_path, "rb") as f_in: 
       iv = f_in.read(16) 
       encrypted = f_in.read() 
   cipher = AES.new(key, AES.MODE_CBC, iv) 
   decrypted_padded = cipher.decrypt(encrypted) 
 
   pad_len = decrypted_padded[-1] 
   decrypted = decrypted_padded[:-pad_len] 
   buffer = io.BytesIO(decrypted) 
   state_dict = torch.load(buffer, 
map_location=torch.device('cpu')) 
   model.load_state_dict(state_dict) 
 
 
######################## 
# Watermark Embedding 
######################## 
def embed_watermark(input_image_path: str, 
watermark_text: str, 
                   output_image_path: str, position=(10, 
10), opacity=128): 
   """ 
   Embeds a simple text watermark onto the input image 
and saves as output_image_path. 
   """ 
   image = Image.open(input_image_path).convert("RGBA") 
   txt_layer = Image.new("RGBA", image.size, (255, 255, 
255, 0)) 
 draw = ImageDraw.Draw(txt_layer) 
 
   try: 
       font = ImageFont.truetype("arial.ttf", 36) 
   except IOError: 
       font = ImageFont.load_default() 
 
   draw.text(position, watermark_text, fill=(255, 255, 
255, opacity), font=font) 
   watermarked = Image.alpha_composite(image, txt_layer) 
   watermarked.convert("RGB").save(output_image_path) 
