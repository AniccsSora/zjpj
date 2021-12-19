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
import numpy as np
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # 暫且安捏
logging.getLogger(__name__)

# --- 超參數
parser = argparse.ArgumentParser("參數設定")
# hyperparameter
parser.add_argument('--epochs', type=int, default=50, help='訓練週期次數')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--drop', type=float, default=0.2, help='conv layer 後的dropout率')
parser.add_argument('--batch_size', type=int, default=32, help='conv layer 後的dropout率')
# optim
parser.add_argument('--reduceLR', type=bool, default=True, help='使用 ReduceLROnPlateau 來優化訓練')
# 權重檔案是否指定
parser.add_argument('--weight_pt', type=str, default="", help='指定權重繼續訓練，留空為重新訓練。')
# 資料夾參數
parser.add_argument('--folder_postfix', type=str, default="", help="資料夾後輟名")
parser.add_argument('--log_dir', type=str, default="log_save", help="存放資料夾名")

param = parser.parse_args()
kwargs = vars(param)  # param 轉成字典 供後方函數使用

# 浮點數轉科學記號
scientific_lr = "{:.0e}".format(param.lr)
title_str = f'lr:{scientific_lr}_drop:{param.drop}_batch:{param.batch_size}'

# --- logger setting
def set_logger(path, log_fname='log.txt'):

    logging.basicConfig(filename=pjoin(path, log_fname), level=logging.INFO, force=True)
    #logging.info(f"command line: {str(sys.argv)}")
    logging.info(f"command line: {' '.join(sys.argv)}")
    logging.info("============ kay args ============")
    for key, val in kwargs.items():
        logging.info("{}, {}".format(key, val))
    logging.info("==================================")
    print("log save:", pjoin(path, log_fname))

def get_min_lr(lr_log):
    return np.amin(np.array(lr_log))


if __name__ == "__main__":
    # --- logger setting
    log_dir = param.log_dir
    current_time = datetime.now()
    fd_postfix = "" if param.folder_postfix == "" else "_"+param.folder_postfix
    save_path = pjoin(log_dir, current_time.strftime('%Y%m%d_%H%M_%S')+fd_postfix)  # 時戳資料夾
    if param.weight_pt != "":
        assert os.path.exists(param.weight_pt)
        save_path += "_reWeighting"  # 表示重新使用舊的 pt
        logging.info(f"繼續訓練 pt: {param.weight_pt}")
        while True:
            _response = str(input("是否繼續訓練 此pt: {}?(y/N)".format(param.weight_pt)))[0]
            if _response == 'y' or _response == 'Y':
                break
            elif _response == 'N':
                param.weight_pt = ""
                break
            else:
                continue
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
    if param.weight_pt != "":
        print("reloading pt:", param.weight_pt)
        net.load_state_dict(torch.load(param.weight_pt))
    _train_start = time.perf_counter()
    net, loss, lr_log = train.train(patch_dataloader, net=net, lr=param.lr,
                            epochs=param.epochs, weight=data_weight,
                            draw=save_path,
                            val_dataloader=val_dataloader, kwargs=kwargs)
    _train_finish = time.perf_counter()

    loss_train = loss['train']
    loss_val = [_.item() for _ in loss['val']]

    torch.save(net.state_dict(), pjoin(save_path, 'weight.pt'))

    # 繪製 loss 圖表
    plt.figure()
    plt.xlim(0, param.epochs)
    plt.xlabel('epoch')
    plt.xticks(np.arange(0, param.epochs, step=5))
    _ylim_max = np.amax(np.array([loss_train, loss_val]).flatten())*1.05
    plt.ylim(0.0, round(_ylim_max+0.05, 2))
    plt.plot(loss_train, label='train')
    plt.plot(loss_val, label='val')
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(save_path, 'loss_graph.png'))

    # 繪製 lr 圖表
    plt.figure()
    plt.xlim(0, param.epochs)
    plt.xlabel('epoch')
    plt.xticks(np.arange(0, param.epochs, step=5))
    plt.ylim(get_min_lr(lr_log)*0.95, param.lr*1.05)
    plt.yscale("log")
    plt.plot(lr_log.tolist(), label='lr')
    plt.title('learning rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(save_path, 'learning_rate.png'))
    plt.close('all')

    logging.info("lr rate dynamic:")
    for lr in lr_log.tolist():
        logging.info(f"  lr: {lr}")
    _cost_t_time = _train_finish - _train_start
    _cost_t_time = str(pd.to_timedelta(timedelta(seconds=_cost_t_time))).split('.')[0]
    print(f"Begin train: {current_time.strftime('%m/%d %H:%M:%S')}")
    print("cost:", _cost_t_time)
    logging.info(f"Begin train: {current_time.strftime('%m/%d %H:%M:%S')}")
    logging.info(f"cost time: {_cost_t_time}")
