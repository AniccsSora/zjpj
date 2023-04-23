import torch
from torch.utils.data import random_split
from data.AugImgDataSet_v2 import AugImgMultiAugDataset

def train(net, dataloader, val_dataloader, epoches, lr, save_root, SAVE_ROUND=1000):
    pass




if __name__ == "__main__":
    augImg_dataset_v2 = AugImgMultiAugDataset(img_root="./data/haha", each_base_name="0_base.png")

    dataloader_v2, val_dataloader_v2, test_dataloader_v2 = random_split(augImg_dataset_v2, [0.7, 0.2, 0.1],
                                                                        generator=torch.Generator().manual_seed(42))

