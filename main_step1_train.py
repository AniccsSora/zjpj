import torch
from dataloader.PatchesDataset import PatchesDataset
from torch.utils.data import DataLoader
from model import QRCode_CNN
import train
import time
from datetime import datetime, timedelta
from os.path import join as pjoin
import os
import matplotlib.pyplot as plt
import argparse
import logging
import pandas as pd
logging.getLogger(__name__)

# --- 超參數
parser = argparse.ArgumentParser("參數設定")
parser.add_argument('--epochs', type=int, default=10, help='訓練週期次數')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--log_dir', type=str, default="log_save", help="存放資料夾名")
parser.add_argument('--drop', type=float, default=0.3, help='conv layer 後的dropout率')
parser.add_argument('--batch_size', type=int, default=32, help='conv layer 後的dropout率')
parser.add_argument('--reduceLR', type=bool, default=False, help='使用 ReduceLROnPlateau 來優化訓練')
# 記錄用參數
parser.add_argument('--folder_postfix', type=str, default="有無動態lr差異測試", help="資料夾後輟名")
param = parser.parse_args()
kwargs = vars(param)  # param 轉成字典 供後方函數使用

# 浮點數轉科學記號
scientific_lr = "{:.0e}".format(param.lr)
title_str = f'lr:{scientific_lr}_drop:{param.drop}_batch:{param.batch_size}'

# --- logger setting
def set_logger(path, log_fname='log.txt'):

    logging.basicConfig(filename=pjoin(path, log_fname), level=logging.INFO, force=True)
    print("log save:", pjoin(path, log_fname))


if __name__ == "__main__":
    # --- logger setting
    log_dir = param.log_dir
    current_time = datetime.now()
    fd_postfix = "" if param.folder_postfix == "" else "_"+param.folder_postfix
    save_path = pjoin(log_dir, current_time.strftime('%Y%m%d_%H%M_%S')+fd_postfix)  # 時戳資料夾
    os.makedirs(save_path)
    set_logger(save_path, log_fname='log.txt')  # 設定 log 存放位置與 log檔名。
    #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"device: {device}")

    # QRCode patch 資料夾
    # QRCode_patch_dir_root = './data/pathes_of_qrcode_32x32'  # 裡面參雜一些很不像 QRCode 的patch
    QRCode_patch_dir_root = './data/manual_pick_QRcode_Patch'

    # 背景 patch 資料夾
    Background_patch_dir_root = './data/background_patch'

    patches_dataset = PatchesDataset(qrcode_patches_dir=QRCode_patch_dir_root,
                                     background_patches_dir=Background_patch_dir_root,
                                     device=device)
    val_dataset = PatchesDataset(qrcode_patches_dir='./data/val_patch_True',
                                 background_patches_dir='./data/val_patch_False',
                                 device=device)

    data_weight = patches_dataset.weight

    logging.info(f"batch_size: {param.batch_size}")
    patch_dataloader = DataLoader(patches_dataset, batch_size=param.batch_size, shuffle=True, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True, pin_memory=False)

    net = QRCode_CNN(drop=param.drop)  # drop: drop layer.
    _train_start = time.perf_counter()
    net, loss, lr_log = train.train(patch_dataloader, net=net, lr=param.lr,
                            epochs=param.epochs, weight=data_weight,
                            draw=save_path,
                            val_dataloader=val_dataloader, kwargs=kwargs)
    _train_finish = time.perf_counter()

    loss_train = loss['train']
    loss_val = [_.item() for _ in loss['val']]

    torch.save(net.state_dict(), pjoin(save_path, 'weight.pt'))

    plt.figure()
    plt.plot(loss_train, label='train')
    plt.plot(loss_val, label='val')
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(save_path, 'loss_graph.png'))

    plt.figure()
    plt.plot(lr_log.tolist(), label='lr')
    plt.title('learning rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(save_path, 'learning_rate.png'))

    _cost_t_time = _train_finish - _train_start
    _cost_t_time = str(pd.to_timedelta(timedelta(seconds=_cost_t_time))).split('.')[0]
    print(f"Begin train: {current_time.strftime('%m/%d %H:%M:%S')}")
    print("cost:", _cost_t_time)
    logging.info(f"Begin train: {current_time.strftime('%m/%d %H:%M:%S')}")
    logging.info(f"cost time: {_cost_t_time}")
