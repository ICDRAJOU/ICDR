import numpy as np
from PIL import Image
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
from utils.image_utils import crop_img

# PSNR과 SSIM 계산 함수
def compute_psnr_ssim(recovered, clean):
    assert recovered.shape == clean.shape, f"Shape mismatch: {recovered.shape} != {clean.shape}"
    recovered = np.clip(recovered / 255.0, 0, 1)  # 정규화
    clean = np.clip(clean / 255.0, 0, 1)  # 정규화

    psnr = peak_signal_noise_ratio(clean, recovered, data_range=1)
    ssim = structural_similarity(clean, recovered, data_range=1, multichannel=True, win_size=3)
    
    return psnr, ssim

class_name = "low_rain"

# 이미지 파일이 있는 디렉토리 경로
clean_dir = './data/CDD-11_test_100/clear/'
restore_dir = [f'./data/CDD-11_test_100/rain/',
'./data/CDD-11_test_100/haze/',
'./data/CDD-11_test_100/low/',
'./data/CDD-11_test_100/haze_rain/',
'./data/CDD-11_test_100/low_rain/',
'./data/CDD-11_test_100/low_haze/',
'./data/CDD-11_test_100/low_haze_rain/',
]

# 디렉토리 내 모든 파일 이름을 가져온 후, 이미지 파일만 필터링
image_files = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

def load_image_as_tensor(image_path):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img) / 255.0  # 0~255 범위를 0~1로 정규화
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    return img

sum_psnr = 0
sum_ssim = 0

# PSNR과 SSIM 계산
for img_file in image_files:
    for restore_dir_list in restore_dir:
        print(img_file)
        restore_img = crop_img(np.array(Image.open(os.path.join(restore_dir_list + img_file)).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(os.path.join(clean_dir + img_file)).convert('RGB')), base=16)
        # print(restore_img.shape, clean_img.shape)
        # if restore_img.shape != clean_img.shape:
            #clean_img = torch.nn.functional.interpolate(clean_img, size=(restore_img.shape[2], restore_img.shape[3]))
        
        psnr, ssim = compute_psnr_ssim(restore_img, clean_img)
        sum_psnr += psnr
        sum_ssim += ssim
        # print(f"{img_file} - PSNR: {psnr}, SSIM: {ssim}")

print(f"PSNR : {sum_psnr/140}, SSIM: {sum_ssim/140}")
