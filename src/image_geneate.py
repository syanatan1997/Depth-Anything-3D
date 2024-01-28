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

# Set Depth Anything
def _set_depth_anything(encoder):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
    
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

    return DEVICE, transform, depth_anything


# Get Image filenames from Input Directory
def get_image_filenames(input_dir):
        filenames = os.listdir(input_dir)
        filenames = [os.path.join(input_dir, filename) for filename in filenames]
        filenames.sort()

        return filenames

# Generate Depth Image from Image files
def generate_depth(filenames, output_dir, encoder):
    depth_paths = []
    DEVICE, transform, depth_anything = _set_depth_anything(encoder)
    
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

        depth_path = os.path.join(output_dir, 'depth_img_' + os.path.basename(filename).split('.')[0] + '.png')
        cv2.imwrite(depth_path, depth)
        depth_paths.append(depth_path)
    
    return depth_paths

# Generate Stereo Right Image from raw image and depth image
def generate_right_image(raw_image, depth, reverse):
    h, w, c = raw_image.shape

    # 深度マップの正規化をベクトル化
    depth_min = depth.min()
    depth_max = depth.max()

    if reverse:
        # Full SBS Reverse
        normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        # Full SBS
        normalized_depth = (depth_max - depth) / (depth_max - depth_min)
    
    #IPD = 6.5
    #MONITOR_W = 38.5
    #deviation_cm = IPD * 0.12
    #deviation = deviation_cm * MONITOR_W * (w / 1920)
    #deviation_pixels = (normalized_depth ** 2) * deviation

    # 偏差の計算
    deviation_pixels = normalized_depth * 30

    # 右画像の生成をベクトル化
    col_indices = np.clip(np.arange(w) - deviation_pixels.astype(int), 0, w - 1)
    right_img = raw_image[np.arange(h)[:, None], col_indices]

    # 空白部分の修正
    mask = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) == 0
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    right_img_fixed = right_img[np.arange(h)[:, None], idx]

    return right_img_fixed

# Combine Raw Image and Right Image to generate Stereo Image
def generate_stereo_image(args):
    filename, depth_path, output_dir, reverse = args
    raw_image = cv2.imread(filename)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    right_image = generate_right_image(raw_image, depth, reverse)
    
    stereo = np.hstack([raw_image, right_image])
    suffix = 'r_sbs_img_' if reverse else 'sbs_img_'
    extension = '.png'
    stereo_image_path = os.path.join(output_dir, suffix + os.path.basename(filename).split('.')[0] + extension)
    cv2.imwrite(stereo_image_path, stereo)       

def generate_all_stereo_images(filenames, depth_paths, output_dir, reverse):
    # プールを作成（プロセッサの数に基づく）
    with Pool() as pool:
        list(tqdm(pool.imap(generate_stereo_image, zip(filenames, depth_paths, [output_dir]*len(filenames), [reverse]*len(filenames),)), total=len(filenames)))

# Generate Stereo Video from video file and output dir
def generate_stereo_video(filename, output_dir, encoder):
    DEVICE, transform, depth_anything = _set_depth_anything(encoder)
    
    raw_video = cv2.VideoCapture(filename)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
    output_width = frame_width * 2
    base_filename = os.path.basename(filename)
    output_path = os.path.join(output_dir, base_filename[:base_filename.rfind('.')] + '_video_depth.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
    
    depth_frames = []  # depth画像を格納するリスト
    raw_image_frames = []  # 元のフレームを格納するリスト

    with tqdm() as pbar:
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            raw_image_frames.append(raw_frame)
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
            
            frame = transform({'image': frame})['image']
            frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                depth = depth_anything(frame)

            depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            
            depth = depth.cpu().numpy().astype(np.uint8)
            depth_frames.append(depth)
            
            pbar.update(1)
        
        print("Depth Finished")
    
    for raw_image_frame, depth_frame in tqdm(zip(raw_image_frames, depth_frames)):
        right_image = generate_right_image(raw_image_frame, depth_frame, False)
        combined_frame = cv2.hconcat([raw_image_frame, right_image])
        out.write(combined_frame)
    print("Stereo finished")
    
    raw_video.release()
    out.release()
    