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
logging.getLogger(__name__)

# --- 超參數
parser = argparse.ArgumentParser("參數設定")
parser.add_argument('--epochs', type=int, default=3, help='訓練週期次數')
parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
parser.add_argument('--log_dir', type=str, default="log_save", help="存放資料夾名")
parser.add_argument('--drop', type=int, default=0.1, help='conv layer 後的dropout率')
parser.add_argument('--batch_size', type=int, default=32, help='conv layer 後的dropout率')

param = parser.parse_args()

# --- logger setting
def set_logger(path, log_fname='log.txt'):

    logging.basicConfig(filename=pjoin(path, log_fname), level=logging.INFO, force=True)
    print("log save:", pjoin(path, log_fname))


if __name__ == "__main__":
    # --- logger setting
    log_dir = param.log_dir
    current_time = datetime.now().strftime('%Y%m%d_%H%M_%S')
    save_path = pjoin(log_dir, current_time)  # 時戳資料夾
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

    net = QRCode_CNN(drop=0.1)  # drop: drop layer.
    _train_start = time.process_time()
    net, loss = train.train(patch_dataloader, net=net, lr=param.lr,
                            epochs=param.epochs, weight=data_weight,
                            draw=save_path,
                            val_dataloader=val_dataloader)
    _train_finish = time.process_time()

    loss_train = loss['train']
    loss_val = [_.item() for _ in loss['val']]

    torch.save(net.state_dict(), pjoin(save_path, 'weight.pt'))

    plt.plot(loss_train, label='train')
    plt.plot(loss_val, label='val')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(save_path, 'loss_graph.png'))

    _cost_t_time = _train_finish - _train_start
    _cost_t_time = str(timedelta(seconds=_cost_t_time))

    print("cost:", _cost_t_time)
    logging.info(f"cost time: {_cost_t_time}")