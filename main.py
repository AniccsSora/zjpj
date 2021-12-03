import torch
from dataloader.PatchesDataset import PatchesDataset
from torch.utils.data import DataLoader
from model import QRCode_CNN
import train
import time
from datetime import datetime
from os.path import join as pjoin
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    patches_dataset = PatchesDataset(qrcode_patches_dir='./data/pathes_of_qrcode_32x32',
                                     background_patches_dir='./data/background_patch',
                                     device=device)
    val_dataset = PatchesDataset(qrcode_patches_dir='./data/val_patch_True',
                                 background_patches_dir='./data/val_patch_False',
                                 device=device)

    data_weight = patches_dataset.weight
    patch_dataloader = DataLoader(patches_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    model_dir = "model_weight"
    net = QRCode_CNN(drop=0.1)  # drop: droplayer

    starttime = time.time()
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    save_path = pjoin(model_dir, current_time)
    os.makedirs(save_path)
    net, loss = train.train(patch_dataloader, net=net, lr=1e-4, epochs=50, weight=data_weight,
                            draw=save_path,
                            val_dataloader=val_dataloader)

    loss_train = loss['train']
    loss_val = [_.item() for _ in loss['val']]

    torch.save(net.state_dict(), f'./model_weight/{current_time}.pt')

    plt.plot(loss_train, label='train')
    plt.plot(loss_val, label='val')
    plt.legend()
