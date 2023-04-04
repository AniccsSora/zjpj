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
from typing import List
from tqdm import tqdm
from misc.F import ensure_folder
import cv2


class NdArrayDataset_RGB(Dataset):
    """
    file_list: 每一張圖片的路徑。

    回傳 np.uint8 型別的 data. (H, W, C=預設 3)
    """
    def __init__(self, file_list: List[Path], transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # 将图像转换为 NumPy 数组，通道数放在第3维
        img = np.array(img).transpose((1, 2, 0))

        if img.dtype == np.float32:
            img = (img * 255.0).astype(np.uint8)

        return img


if __name__ == "__main__":
    DATASET_PATH = Path("./perspective_qrCodes")
    fn = [_ for _ in DATASET_PATH.glob("*.*")]

    # 定義 transformer
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # __getitem__ 回的就是 RGB 了。
        #transforms.Lambda(lambda x: x.convert('RGB')),  # 將圖片轉換為 RGB 格式
        #transforms.Lambda(lambda x: x.convert('L')),  # 將圖片轉換為灰階
        transforms.ToTensor(),
    ])

    #
    my_dataset = NdArrayDataset_RGB(fn, transform=transform)

    # rotate_F_aug 內的 name
    # 會變成 sub_dir 的後綴名。
    rotate_F_aug = [
        iaa.Affine(rotate=0, name="0"),
        iaa.Affine(rotate=90, name="90"),
        iaa.Affine(rotate=180, name="180"),
        iaa.Affine(rotate=270, name="270"),
        iaa.Sequential([iaa.Fliplr(1)
                        ], name="F"),
        iaa.Sequential([iaa.Affine(rotate=90),
                        iaa.Fliplr(1)], name="F90"),
        iaa.Sequential([iaa.Affine(rotate=180),
                        iaa.Fliplr(1)], name="F180"),
        iaa.Sequential([iaa.Affine(rotate=270),
                        iaa.Fliplr(1)], name="F270"),
    ]
    # 每個都要有名稱，因為創立子資料夾名稱會用到
    for _ in rotate_F_aug:
        assert _.name

    # ===================================================
    # 這個東西就自己拚，但須注意的是不需用去 (旋轉/翻轉) 他。
    # ===================================================
    aug_loop = [
        #
        iaa.Sequential([
            iaa.Add((-40, 40), per_channel=True),
        ], name="Add"),
        #
        iaa.AdditiveGaussianNoise(scale=(0, 0.2*255), per_channel=True, name="GaussN"),
        #
        iaa.AdditivePoissonNoise(40, per_channel=0.5, name="PoissonN40"),
        #
        iaa.Multiply((0.5, 1.5), per_channel=0.5, name="Multiply"),
        #
        iaa.Cutout(nb_iterations=2,
                   fill_mode="constant",
                   cval=(0, 255),
                   fill_per_channel=0.5,
                   squared=False,
                   size=0.15,
                   name="ComplexCutout"),
        #
        iaa.Cutout(nb_iterations=4, size=0.1, fill_per_channel=0.5, squared=False, name="Cutout"),
        #
        iaa.Dropout(p=(0, 0.2), per_channel=0.5, name="Dropout"),
        #
        # 第一個參數是丟棄整張圖的 ?% 的點
        iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5, name="CoarseDropout"),
        #
        iaa.Dropout2d(p=0.5, name="DropOutHalf"),
        #
        iaa.ImpulseNoise(p=(0.1, 0.3), name="ImpulseNoise"),
        #
        iaa.SaltAndPepper(p=(0.1, 0.3), name="SaltPepper"),
        #
    ]

    #
    save_path = Path("./haha")
    #
    #
    SUB_DIR_ZFILL_LENGTH = 6
    #
    # 每個 aug_loop 內要跑幾次
    REPEAT_AUG_TIMES = 2
    #
    # root save path
    ensure_folder(save_path, remake=True)
    #
    # 計算 成長倍數，and check
    aug_times = len(rotate_F_aug) * len(aug_loop) * REPEAT_AUG_TIMES
    print("計算後 強化倍數:", aug_times)
    print(f"圖片增長為 : {len(my_dataset)} >> {len(my_dataset) * aug_times}...")
    while True:
        user_input = input("同意後按下 Y to continue (N 終止): ")
        if user_input.lower() == "y":
            break
        if user_input.lower() == "n":
            exit(88)
    #
    #
    pbar = tqdm(my_dataset)
    for idx, img in enumerate(pbar):
        if idx > 10: # for test purpose
            print("go break...")
            break
        # process
        pbar.set_description(f"Processing {idx}: ")

        # 旋轉的
        for _rotate_auger in rotate_F_aug:
            # 根據 rotate_F_aug 去轉照片，後續的處理根據這張圖片做事
            _r_img = _rotate_auger.augment_image(img)

            # 建立初步的 子資料夾名稱，這之後會使用
            _sub_dir = save_path.joinpath(str(idx).zfill(SUB_DIR_ZFILL_LENGTH))
            #
            # 附上 旋轉後綴名的 資料夾名稱
            sub_dir = Path(f"{_sub_dir}_{_rotate_auger.name}")
            ensure_folder(sub_dir, remake=True)

            # 在該 子資料夾下存放 原始旋轉後照片。
            # 這張圖片即為 label (原始的照片)。
            _r_img_RGB = cv2.cvtColor(_r_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(sub_dir.joinpath(f"0_base.png").__str__(), _r_img_RGB)

            # save aug
            for auger in aug_loop:
                for aug_times in range(1, REPEAT_AUG_TIMES+1):
                    _aug_img = auger.augment_image(_r_img)
                    _r_img_RGB = cv2.cvtColor(_aug_img, cv2.COLOR_BGR2RGB)
                    if auger.name:
                        _auged_save_path = sub_dir.joinpath(f"{sub_dir.name}_{auger.name}_{aug_times}.png").__str__()
                    else:
                        _auged_save_path = sub_dir.joinpath(f"{sub_dir.name}_{aug_times}.png").__str__()
                    # 強化過後的 儲存路徑。
                    cv2.imwrite(_auged_save_path, _r_img_RGB)
