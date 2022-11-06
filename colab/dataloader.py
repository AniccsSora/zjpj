import torch
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from torch import nn
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import os
import tqdm
import util as mutil

class QR_good_bad_dataset(Dataset):
    """
    bad_qr and good_qr
    """
    def __init__(self, data_path, image_size):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.samples = self.init()

    def init(self):
        """
        scan good and bad folder below data_path。。。
        """
        good = [_ for _ in self.data_path.glob("good/*.*")]
        bad = [_ for _ in self.data_path.glob("bad/*.*")]
        assert len(good) == len(bad)  # good bad 資料夾下資料有不同數量的資料。
        res = []
        for i in range(len(good)):
            if (good[i].stem == bad[i].stem):
                tmp = (good[i], bad[i])
                res.append(tmp)
        assert len(res) > 0  # 沒有資料。。。
        return res

    def __getitem__(self, index):
        good_path, bad_path = self.samples[index]

        # read file
        g_img = cv2.imread(good_path.__str__(), cv2.IMREAD_GRAYSCALE)
        #g_img = cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB)
        b_img = cv2.imread(bad_path.__str__(), cv2.IMREAD_GRAYSCALE)
        #b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)

        # make square
        g_img = mutil.pad_2_square(g_img, self.image_size)
        b_img = mutil.pad_2_square(b_img, self.image_size)

        # convert to torch-style shape
        # [h, w, channel] --> [channel, h, w]

        g_img = torch.tensor(g_img)
        # 1 是 channel
        g_img = torch.reshape(g_img, shape=(self.image_size, self.image_size, 1))
        g_img = torch.permute(g_img, (2, 0, 1))
        b_img = torch.tensor(b_img)
        b_img = torch.reshape(b_img, shape=(self.image_size, self.image_size, 1))
        b_img = torch.permute(b_img, (2, 0, 1))

        return torch.clone(g_img), torch.clone(b_img)

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    my_dataset = QR_good_bad_dataset("./data", image_size=128)

    my_dataloader = DataLoader(my_dataset, batch_size=32, drop_last=False, pin_memory=False)

    for i in my_dataloader:
        g, b = i
        print(g.size())
        print(b.size())

    pass
