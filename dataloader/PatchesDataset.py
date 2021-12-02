import torch
from torch.utils.data import Dataset
import glob
from os.path import join as pjoin
from PIL import Image
import cv2
import numpy as np
import random
import os

class PatchesDataset(Dataset):
    """
    32x32 patches
    """
    def __init__(self, qrcode_patches_dir, background_patches_dir, device):
        if qrcode_patches_dir is not None:
            assert os.path.isdir(qrcode_patches_dir)  # 請確認 qrcode patch 資料夾存在
            self.qr_patch_path_list = glob.glob(pjoin(qrcode_patches_dir, '*/*.*'), recursive=True)
        else:
            self.qr_patch_path_list = []
        if background_patches_dir is not None:
            assert os.path.isdir(background_patches_dir)  # 請確認 background patches 資料夾存在
            self.background_patch_path_list = glob.glob(pjoin(background_patches_dir, '*/*.*'), recursive=True)
        else:
            self.background_patch_path_list = []
        self.device = device
        self.data = self.qr_patch_path_list + self.background_patch_path_list
        qr_ratio = len(self.qr_patch_path_list) / len(self.data)
        bg_ratio = len(self.background_patch_path_list) / len(self.data)
        self.weight = {'background': bg_ratio, 'QRCode': qr_ratio}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        is_qrcode = False
        if idx < len(self.qr_patch_path_list):
            is_qrcode = True
        image = cv2.imread(self.data[idx], cv2.IMREAD_GRAYSCALE)
        label = 1 if is_qrcode else 0

        image = image/255.0
        res_tensor = torch.tensor(image, device=self.device, dtype=torch.float32)

        assert len(res_tensor.shape) == 2
        if res_tensor.shape != (32, 32):
            res_tensor = self.resize_32_32(res_tensor)
        assert res_tensor.shape == (32, 32)

        # return torch.unsqueeze(res_tensor, 0), label
        return res_tensor, label

    def resize_32_32(self, data):
        res = None
        data = data.detach().cpu().numpy()
        if data.shape != (32, 32):
            res = cv2.resize(data, dsize=(32, 32))
        else:
            res = data

        return torch.tensor(res).to(self.device)
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patches_dataset = PatchesDataset(qrcode_patches_dir='../data/pathes_of_qrcode_32x32',
                                      background_patches_dir='../data/background_patch',
                                     device=device)

    print("資料比例:", patches_dataset.weight)
    while True:
        _ = random.randint(0, len(patches_dataset))
        image, label = patches_dataset[_][0].cpu().detach().numpy(), patches_dataset[_][1]
        label_mean = "QR Code" if label == 1 else "Background"
        cv2.imshow(f"{label}: {label_mean}", np.array(Image.fromarray(image).resize((320, 320))))
        cv2.waitKey(1)