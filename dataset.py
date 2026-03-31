import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap
    print("✅ NuScenes + Map API loaded successfully")
except ImportError:
    NuScenes = None
    NuScenesMap = None

try:
    from pyquaternion import Quaternion
except ImportError:
    Quaternion = None


class NuScenesDrivableDataset(Dataset):
    def __init__(self, dataroot, version='v1.0-mini', target_size=(512, 256)):
        self.dataroot = dataroot
        self.version = version
        self.target_w, self.target_h = target_size

        print(f"Loading NuScenes version {version} from {dataroot}...")

        if NuScenes is None:
            raise ImportError("NuScenes not installed")

        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        self.samples = []

        for scene in self.nusc.scene:
            token = scene['first_sample_token']
            while token != "":
                sample = self.nusc.get('sample', token)
                self.samples.append(sample['data']['CAM_FRONT'])
                token = sample['next']

        print(f"✅ Total CAM_FRONT samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    # 🔥 NEW: SYNTHETIC DRIVABLE MASK (CRITICAL FIX)
    def generate_synthetic_mask(self, img):
        h, w, _ = img.shape

        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # edge detection
        edges = cv2.Canny(gray, 50, 150)

        # road usually bottom region
        mask = np.zeros((h, w), dtype=np.uint8)

        pts = np.array([
            [int(0.1*w), h],
            [int(0.9*w), h],
            [int(0.6*w), int(0.6*h)],
            [int(0.4*w), int(0.6*h)]
        ], np.int32)

        cv2.fillPoly(mask, [pts], 1)

        # refine using edges
        mask = cv2.bitwise_and(mask, (edges > 0).astype(np.uint8) + 1)

        return mask

    def __getitem__(self, idx):
        cam_token = self.samples[idx]
        sd_record = self.nusc.get('sample_data', cam_token)

        img_path = os.path.join(self.dataroot, sd_record['filename'])
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.target_w, self.target_h))

        img_np = np.array(img)

        # 🔥 ALWAYS USE SYNTHETIC MASK
        mask = self.generate_synthetic_mask(img_np)

        # HSV proxy
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        v_chan = hsv[:, :, 2]

        v_edges = cv2.Canny(v_chan, 50, 150)
        hsv_v_proxy = (v_edges > 0).astype(np.float32)

        # condition label
        if np.mean(v_chan) < 40:
            condition = 3
        elif np.std(v_chan) < 50:
            condition = 2
        elif np.mean(v_chan) > 200:
            condition = 4
        else:
            condition = 0

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()
        hsv_v_proxy_tensor = torch.from_numpy(hsv_v_proxy)
        condition_tensor = torch.tensor(condition)

        return img_tensor, mask_tensor, hsv_v_proxy_tensor, condition_tensor