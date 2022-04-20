import torch
from torch.utils.data import Dataset, DataLoader
import glob
from os.path import join as pjoin
from PIL import Image
import cv2
import numpy as np
import random
import os
import time
from torchvision.datasets import FakeData
import torchvision
import matplotlib.pyplot as plt

class PatchesDataset(Dataset):
    """
    32x32 patches
    """
    def __init__(self, qrcode_dir_list, background_dir_list, device):
        self.qr_patch_path_list = self.find_all_file(qrcode_dir_list)
        self.background_patch_path_list = self.find_all_file(background_dir_list)
        self.device = device
        self.data = self.qr_patch_path_list + self.background_patch_path_list
        qr_ratio = len(self.qr_patch_path_list) / len(self.data)
        bg_ratio = len(self.background_patch_path_list) / len(self.data)
        self.weight = {'background': bg_ratio, 'QRCode': qr_ratio}

    def find_all_file(self, path_list):
        res_path_list = None
        if path_list is not None:
            _ = None
            for dir in path_list:
                assert os.path.isdir(dir)  # 請確認 qrcode patch 資料夾存在
                _ = glob.glob(pjoin(dir, '*/*.*'), recursive=True)
                if res_path_list is None:
                    res_path_list = _
                else:
                    res_path_list = res_path_list + _
        else:
            res_path_list = []
        return res_path_list

    def __len__(self):
        return len(self.data)

    def make_square(self, img):
        h, w = img.shape[0], img.shape[1]
        if w == h:  # 長寬相等直接返回
            return img
        # 將長寬比不為 1:1 的套用 padding 只補單邊補齊，隨機補。
        diff = abs(w-h)
        rd_tk = random.randint(0, 100) % 2 == 0  # 左補右捕的 token
        if w > h:
            # 補高
            if rd_tk:
                pad_width = ((diff, 0), (0, 0))
            else:
                pad_width = ((0, diff), (0, 0))
        else:
            # 補寬
            if rd_tk:
                pad_width = ((0, 0), (0, diff))
            else:
                pad_width = ((0, 0), (diff, 0))

        img = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)

        return img

    def __getitem__(self, idx):
        is_qrcode = False
        if idx < len(self.qr_patch_path_list):
            is_qrcode = True
        image = cv2.imread(self.data[idx], cv2.IMREAD_GRAYSCALE)
        label = 1 if is_qrcode else 0
        label = torch.tensor(label, device=self.device, dtype=torch.uint8)

        image = self.make_square(image)

        ## 模糊化
        image = cv2.GaussianBlur(image, (11, 11), 3)
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


def get_Dataloader(qrcode_dir=None, background_dir=None, device=None,
                   batch_size=64, shuffle=True):
    """

    @param qrcode_dir: 與 background_dir 可以擇一設定或都指定。
    @param background_dir: 與 qrcode_dir 可以擇一設定或都指定。
    @param device: 可不指定。
    @param batch_size: 預設 64。
    @param shuffle: True
    @return: torch.util.data.Dataloader
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not isinstance(qrcode_dir, list):
        qrcode_dir = [qrcode_dir]
    if not isinstance(background_dir, list):
        background_dir = [background_dir]

    patches_dataset = PatchesDataset(qrcode_dir_list=qrcode_dir,
                                     background_dir_list=background_dir,
                                     device=device)

    res_dataloader = DataLoader(patches_dataset,
                                batch_size=batch_size, shuffle=shuffle)
    return res_dataloader


if __name__ == "__main__":
    # 測試 假數據 dataset

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    qrcode_dir_list = [r'D:\Git\zjpj\data_clean\the_real593_patches\filter_OK']
    background_dir_list = ['../data/background_patch']
    patches_dataset = PatchesDataset(qrcode_dir_list,
                                     background_dir_list,
                                     device=device)

    print("資料比例:", patches_dataset.weight)

    for idx in patches_dataset:
        pass

