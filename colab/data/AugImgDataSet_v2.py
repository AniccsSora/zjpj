from torchvision import transforms
import imgaug.augmenters as iaa
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from misc.F import ensure_folder
import cv2
from torch.utils.data import random_split


# 定義數據集類別
class AugImgMultiAugDataset(Dataset):
    def __init__(self, img_root, each_base_name):
        """
        @param img_root:
        @param each_base_name: 每個子資料夾的圖片檔名，每個資料夾都會有一個 base (原始)，與其他(多張)被強化過的圖片，
                                而 此參數指定 唯一 base 名稱。
        """
        self.root = Path(img_root)
        self.base = each_base_name
        #
        self.data_origin = []  # 原始圖
        self.data_augImg = []  # 強化後的

        # check
        assert self.root.is_dir()
        print(f"{self.root} exists... OK")

        pbar = tqdm(self.root.glob("*"))
        # check
        for idx, sub_dir in enumerate(pbar):
            pbar.set_description(f"Processing {idx}: ")
            _all_file_in_sub_dir = [_ for _ in sub_dir.glob("*")]
            _origin_img_name = sub_dir.joinpath(self.base)
            # 存在 self.base 檔案
            assert _origin_img_name.is_file()
            # self.base 在 sub_dir 內, pop() 不存在會跳例外。
            assert _origin_img_name in _all_file_in_sub_dir

            # 存放原始圖片的 path
            self.data_origin.append(_origin_img_name)

            # 存放強化後的其他強化過後圖片的 path
            _all_file_in_sub_dir.remove(_origin_img_name)
            # !!!! 這邊的 _all_file_in_sub_dir 已被移除原始圖片 !!!!
            self.data_augImg.append([_all_file_in_sub_dir])

        # finally check
        #
        assert len(self.data_origin) == len(self.data_augImg)

    def __len__(self):
        return len(self.data_origin)

    def __getitem__(self, index) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        @param index:
        @return: x,y = (原始圖, 強化過後的 list)
        """
        # to tensor f
        transform2tensor = transforms.ToTensor()

        _xStr = self.data_origin[index]
        _yStrList = self.data_augImg[index][0]

        # x 只是一張圖
        x = Image.open(_xStr).convert("RGB")
        x = transform2tensor(x)
        #x = torch.permute(x, (0, 1, 2))

        y = []
        # y 是一個 tensor list
        for yPath in _yStrList:
            y_tmp = Image.open(yPath).convert("RGB")
            y_tmp = transform2tensor(y_tmp)
            #y_tmp = torch.permute(y_tmp, (0, 1, 2))
            y.append(y_tmp)

        return x, y


if __name__ == "__main__":
    # 創建數據集

    augImg_dataset_v2 = AugImgMultiAugDataset(img_root="./haha", each_base_name="0_base.png")

    x, y = augImg_dataset_v2[777]
    # x image debug
    plt.imshow(torch.permute(x, (1, 2, 0)))
    # plt.show()
    # y image debug, y index base on 強化幾張圖片 0 ~ N-1
    plt.imshow(torch.permute(y[0], (1, 2, 0)))
    # plt.show()

    # 分 dataloader
    dataloader_v2, val_dataloader_v2, test_dataloader_v2 = random_split(augImg_dataset_v2, [0.7, 0.2, 0.1],
                                                                  generator=torch.Generator().manual_seed(42))

    # 創建 DataLoader
    #dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print("End")