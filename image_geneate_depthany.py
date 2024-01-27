from multiprocessing import Pool
import cupy as cp
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


def generate_depth(filenames, outdir):
    encoder = 'vitl'
    depth_paths = []

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'models\depth_anything_vitl14'
    
    depth_anything = DepthAnything.from_pretrained(model_path.format(encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = depth_anything(image)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth = depth.cpu().numpy().astype(np.uint8)
        # depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        depth_path = os.path.join(outdir, 'depth_img_' + os.path.basename(filename).split('.')[0] + '.png')
        cv2.imwrite(depth_path, depth)
        depth_paths.append(depth_path)
    
    return depth_paths

def generate_stereo_right(left_img, depth, reverse):
    IPD = 6.5
    MONITOR_W = 38.5
    h, w, c = left_img.shape

    # 深度マップの正規化をベクトル化
    depth_min = depth.min()
    depth_max = depth.max()

    if reverse:
        # Full SBS Reverse
        normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        # Full SBS
        normalized_depth = (depth_max - depth) / (depth_max - depth_min)
    
    # 偏差の計算
    deviation_cm = IPD * 0.12
    deviation = deviation_cm * MONITOR_W * (w / 1920)
    deviation_pixels = (normalized_depth ** 2) * deviation

    # 右画像の生成をベクトル化
    col_indices = np.clip(np.arange(w) - deviation_pixels.astype(int), 0, w - 1)
    right_img = left_img[np.arange(h)[:, None], col_indices]

    # 空白部分の修正
    mask = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) == 0
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    right_img_fixed = right_img[np.arange(h)[:, None], idx]

    return right_img_fixed

def generate_stereo_image(args):
    filename, depth_path, outdir, reverse = args
    raw_image = cv2.imread(filename)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    right_image = generate_stereo_right(raw_image, depth, reverse)
    
    stereo = np.hstack([raw_image, right_image])
    suffix = 'r_sbs_img_' if reverse else 'sbs_img_'
    extension = '.png'
    stereo_image_path = os.path.join(outdir, suffix + os.path.basename(filename).split('.')[0] + extension)
    cv2.imwrite(stereo_image_path, stereo)       

def generate_all_stereo_images(filenames, depth_paths, outdir, reverse):
    # プールを作成（プロセッサの数に基づく）
    with Pool() as pool:
        list(tqdm(pool.imap(generate_stereo_image, zip(filenames, depth_paths, [outdir]*len(filenames), [reverse]*len(filenames),)), total=len(filenames)))

def  main():
    img_path = "assets\input"
    outdir = "assets\output"
    reverse = False

    if os.path.isfile(img_path):
        if img_path.endswith('txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        filenames = os.listdir(img_path)
        filenames = [os.path.join(img_path, filename) for filename in filenames]
        filenames.sort()
    

    depth_paths = generate_depth(filenames, outdir)
    generate_all_stereo_images(filenames, depth_paths, outdir, reverse)    
        
        
'''   
if __name__ == "__main__":
    main()
'''  