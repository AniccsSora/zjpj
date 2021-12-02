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
        assert os.path.isdir(qrcode_patches_dir)  # 請確認 qrcode patch 資料夾存在
        assert os.path.isdir(background_patches_dir)  # 請確認 background patches 資料夾存在
        self.device = device
        self.qr_patch_path_list = glob.glob(pjoin(qrcode_patches_dir, '*/*.*'), recursive=True)
        self.background_patch_path_list = glob.glob(pjoin(background_patches_dir, '*/*.*'), recursive=True)
        self.data = self.qr_patch_path_list + self.background_patch_path_list
        self.weight = (len(self.qr_patch_path_list)/len(self.data), len(self.background_patch_path_list)/len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        is_qrcode = False
        if idx < len(self.qr_patch_path_list):
            is_qrcode = True
        image = cv2.imread(self.data[idx], cv2.IMREAD_GRAYSCALE)
        label = 1 if is_qrcode else 0

        image = image/255
        res_tensor = torch.tensor(image, device=self.device, dtype=torch.float32)

        assert len(res_tensor.shape) == 2
        return torch.unsqueeze(res_tensor, 0), label

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