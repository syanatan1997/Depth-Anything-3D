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

import random
import string

class Utils:
    def __init__(self) -> None:
        pass
    
    def generate_random_string(self, length):
        # 英字と数字を含む文字列を定義
        characters = string.ascii_letters + string.digits
        # 指定された長さのランダムな文字列を生成
        random_string = ''.join(random.choice(characters) for _ in range(length))
        return random_string

class SetGenerate:
    def __init__(self) -> None:
        pass
    
    # Set Depth Anything
    def set_depth_anything(self, encoder):
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

        self.device = DEVICE
        self.transform = transform
        self.depth_anything = depth_anything

    def set_reverse(self, reverse):
        self.reverse = reverse

    def set_output_directory(self, output_dir):
        self.output_dir = output_dir

    # Get Image filenames from Input Directory
    def set_image_filenames(self, input_dir):
            image_filenames = os.listdir(input_dir)
            image_filenames = [os.path.join(input_dir, filename) for filename in image_filenames]
            image_filenames.sort()

            self.image_filenames = image_filenames

    def set_video_filename(self, video_filename):
        self.video_filename = video_filename


class ImageGenerate(SetGenerate):
    def __init__(self) -> None:
        super().__init__()

    # Generate Depth Image from Image files
    def generate_depth_image(self, raw_image):
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            
        h, w = image.shape[:2]
            
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
            
        with torch.no_grad():
            depth = self.depth_anything(image)
            
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            
        depth = depth.cpu().numpy().astype(np.uint8)

        return depth
    
    # Generate Stereo Right Image from raw image and depth image
    def generate_right_image(self, raw_image, depth):
        # cuPyを使用して画像と深度マップをGPUに転送
        raw_image_gpu = cp.asarray(raw_image)
        depth_gpu = cp.asarray(depth)
        
        h, w, c = raw_image_gpu.shape

        # 深度マップの正規化をベクトル化
        depth_min = depth_gpu.min()
        depth_max = depth_gpu.max()

        if self.reverse:
            normalized_depth = (depth_gpu - depth_min) / (depth_max - depth_min)
        else:
            normalized_depth = (depth_max - depth_gpu) / (depth_max - depth_min)

        # 偏差の計算
        deviation_pixels = normalized_depth * 30

        # 右画像の生成をベクトル化
        col_indices = cp.clip(cp.arange(w) - deviation_pixels.astype(cp.int32), 0, w - 1)
        right_img_gpu = raw_image_gpu[cp.arange(h)[:, None], col_indices]

        # 空白部分の修正はOpenCVを利用しているため、GPU上での処理には適さない部分があります。
        # この部分はCPUにデータを戻してから処理を行うか、またはcuPyの機能を使って同様の処理を実装する必要があります。
        # ここでは簡単のため、GPU->CPUに転送してからOpenCVで処理します。
        right_img = cp.asnumpy(right_img_gpu)
        #mask = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) == 0
        #idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        #np.maximum.accumulate(idx, axis=1, out=idx)
        #right_img_fixed = right_img[np.arange(h)[:, None], idx]

        return right_img



    # Combine Raw Image and Right Image to generate Stereo Image
    def generate_stereo_image(self):
        for image_filename in tqdm(self.image_filenames, desc="Generating stereo images"):
            raw_image = cv2.imread(image_filename)

            depth_image = ImageGenerate.generate_depth_image(self, raw_image)
            right_image = ImageGenerate.generate_right_image(self, raw_image, depth_image)
            
            stereo = cv2.hconcat([raw_image, right_image])
           
            suffix_stereo = 'r_sbs_img_' if self.reverse else 'sbs_img_'
            suffix_depth = 'depth_img_'
            extension = '.png'
            depth_image_path = os.path.join(self.output_dir, suffix_depth + os.path.basename(image_filename).split('.')[0] + extension)
            stereo_image_path = os.path.join(self.output_dir, suffix_stereo + os.path.basename(image_filename).split('.')[0] + extension)

            #cv2.imwrite(depth_image_path, depth_image)
            cv2.imwrite(stereo_image_path, stereo)

    

class VideoGenerate(ImageGenerate):
    def __init__(self) -> None:
        super().__init__()

    # Generate Stereo Video from video file and output dir
    def generate_stereo_video(self):
        raw_video = cv2.VideoCapture(self.video_filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2
        base_filename = os.path.basename(self.video_filename)
        output_path = os.path.join(self.output_dir, base_filename[:base_filename.rfind('.')] + '_video_depth.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))

        with tqdm() as pbar:
            while raw_video.isOpened():
                ret, raw_frame = raw_video.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
                
                frame = self.transform({'image': frame})['image']
                frame = torch.from_numpy(frame).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    depth = self.depth_anything(frame)

                depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                
                depth_frame = depth.cpu().numpy().astype(np.uint8)

                right_image = VideoGenerate.generate_right_image(self, raw_frame, depth_frame)
                combined_frame = cv2.hconcat([raw_frame, right_image])
                out.write(combined_frame)
                
                pbar.update(1)
            
            
        print("Stereo Video Encode Finished")
        raw_video.release()
        out.release()
    



    '''
    from multiprocessing import Pool

    def generate_all_stereo_images(filenames, depth_paths, output_dir, reverse):
        # プールを作成（プロセッサの数に基づく）
        with Pool() as pool:
            list(tqdm(pool.imap(generate_stereo_image, zip(filenames, depth_paths, [output_dir]*len(filenames), [reverse]*len(filenames),)), total=len(filenames)))
    '''