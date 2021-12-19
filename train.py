import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader.PatchesDataset import PatchesDataset, get_Dataloader
from torch.utils.data import DataLoader
from model import QRCode_CNN
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from os.path import join as pjoin
import os
logging.getLogger(__name__)
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def param_foolproof_1(draw, val_dataloader):
    if val_dataloader is None:
        assert draw is None
    if draw is None:
        assert val_dataloader is None


def _useless_step():
    class Useless:
        def __init__(self):
            pass

        def step(self, qq):
            pass
    return Useless()


def get_ReduceLROnPlateau(optimizer):
    """
    @note:
    mode(str) {'min', 'max'}: 在'min' mode中,當監控數值"停止下降"時，lr將會降低。
                              反之 'max'監控數值"停止上升"時lr下降。default: 'min'
    """
    res = ReduceLROnPlateau(optimizer,
                            mode='min',
                            factor=0.1,  # new_lr = old_lr * factor 的意思。default=0.1
                            patience=5,  # 可以忍受幾個 eopches 不下降
                            verbose=True,  # 下降時是否提示
                            threshold=1e-4,  # 小數點第幾位當成變化閥值?default: 1e-4
                            threshold_mode='rel',  # default: rel
                            min_lr=1e-6,  # 最小的學習率。default=1e-4
                            cooldown=0,  # 觸發更新條件後，等待幾個epoches 再監視。default=0
                            eps=1e-8  # 更新前後 的 lr 差距小於此值時候，不更新此次的 lr。default=1e-8
                            )
    return res

def get_min_lr(lr_log):
    return np.amin(np.array(lr_log))

def draw_loss_and_lr_figure(epochs, lr, loss_train_and_val: list, title_str, save_path, lr_log):
    assert len(loss_train_and_val) == 2
    loss_train, loss_val = loss_train_and_val[0], loss_train_and_val[1]
    # 繪製 loss 圖表
    plt.figure()
    plt.xlim(0, epochs)
    plt.xlabel('epoch')
    plt.xticks(np.arange(0, epochs, step=epochs//10))
    _ylim_max = np.amax(np.array([loss_train, loss_val]).flatten()) * 1.05
    plt.ylim(0.0, round(_ylim_max + 0.05, 2))
    plt.plot(loss_train, label='train')
    plt.plot(loss_val, label='val')
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(save_path, 'loss_graph.png'))
    plt.close('all')

    # 繪製 lr 圖表
    plt.figure()
    plt.xlim(0, epochs)
    plt.xlabel('epoch')
    plt.xticks(np.arange(0, epochs, step=epochs//10))
    plt.ylim(get_min_lr(lr_log) * 0.95, lr * 1.05)
    plt.yscale("log")
    plt.plot(lr_log, label='lr')
    plt.title('learning rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(save_path, 'learning_rate.png'))
    plt.close('all')

def train(dataloader, net, lr, epochs, weight=None,
          class_name=['background', 'QRCode'],
          draw=None,
          val_dataloader=None,
          kwargs=None):
    assert kwargs is not None  # 必須解析參數
    # -----------------------
    # 訓練時的各種旗標
    use_reduceLR = None  # 是否在訓練時期動態降低 lr。
    # -----------------------
    if kwargs is not None:
        use_reduceLR = kwargs['reduceLR']
    # ------------------------
    # 參數防呆
    param_foolproof_1(draw, val_dataloader)
    # logging
    logging.info("lr: {}".format(lr))
    logging.info("epochs: {}".format(epochs))

    if torch.cuda.is_available():
        net.cuda()
    if weight is not None:
        print("Dataset weight:", weight)
        logging.info("\rUse dataset weight to train.")
        logging.info("\rDataset weight: {}".format(weight))
        # 注意: 跟label相反，label:0 -> 背景， label:1 -> QRCode，所以權重將是相反。
        QR_code_weight = 1.0
        background_weight = 1.0
        weight = torch.tensor([weight['QRCode'] * QR_code_weight,
                               weight['background'] * background_weight
                               ]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    if use_reduceLR:
        scheduler = get_ReduceLROnPlateau(optimizer)
    else:
        scheduler = _useless_step()

    eval_loss = {
        "train": [],
        "val": []
    }
    logging.info("===================================")
    lr_log = []
    for epoch in range(epochs):
        running_loss_in_epoch, cnt = 0, 0
        for data in tqdm(dataloader, desc=f'Epoch {epoch+1} - Training', position=0, leave=True):
            train_images, train_labels = data
            optimizer.zero_grad()
            y_pred = net(train_images)
            loss = criterion(y_pred, train_labels)
            loss.backward()
            optimizer.step()
            # ---
            running_loss_in_epoch += loss.item()
            cnt += 1
        torch.save(net.state_dict(), f"{draw}/weight_{epoch+1}.pt")
        if os.path.exists(f"{draw}/weight_{epoch}.pt"):
            os.remove(f"{draw}/weight_{epoch}.pt")
        if val_dataloader is not None:
            with torch.no_grad():
                val_total_loss = 0.0
                val_cnt = 0
                for val_data in tqdm(val_dataloader, desc=f'\tEpoch {epoch+1} - Evaluating', leave=True):
                    val_images, val_labels = val_data
                    val_y_pred = net(val_images)
                    val_total_loss += criterion(val_y_pred.to(device), val_labels.to(device))
                    val_cnt += 1
                _ = val_total_loss / val_cnt
                eval_loss['val'].append(_)
                print("\t val loss:", _.item())
        # end of validation.
        train_loss = running_loss_in_epoch / cnt
        eval_loss['train'].append(train_loss)
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'\repoch: {epoch + 1}, loss: {train_loss}, lr: {current_lr}')
        logging.info(f'epoch: {epoch + 1}, loss: {train_loss}, lr: {current_lr}')
        lr_log.append(current_lr)
        # draw loss figure
        loss_train_and_val = [eval_loss['train'], [_.item() for _ in eval_loss['val']]]
        draw_loss_and_lr_figure(epochs, lr, loss_train_and_val,
                                "in_training", draw, lr_log)
        scheduler.step(running_loss_in_epoch)
    return net, eval_loss, np.array(lr_log)


if __name__ == "__main__":
    # 參數
    kwargs = dict()
    kwargs['reduceLR'] = True
    patches_dataset = PatchesDataset(qrcode_patches_dir='./data/pathes_of_qrcode_32x32',
                                     background_patches_dir='./data/background_patch',
                                     device=device)

    train_dataloader = DataLoader(patches_dataset, batch_size=64, shuffle=True)

    # test
    tdata = next(iter(train_dataloader))
    images, labels = tdata

    net = QRCode_CNN(drop=0.1)

    net, eval_loss, lr_log = train(train_dataloader, net, lr=1e-4, epochs=2,
                                   weight=patches_dataset.weight,
                                   kwargs=kwargs)
    torch.save(net.state_dict(), './trained_e4_ep50.pt')
    val_background_dir="./data/val_patch_False"
    val_qrcode_dir = "./data/val_patch_True"
    same_qr_dir = './data/pathes_of_qrcode_32x32'
    same_bk_dir = './data/background_patch'

    val_qr_dataset = PatchesDataset(qrcode_patches_dir=val_qrcode_dir,
                                    background_patches_dir=None,
                                    device=device)
    val_bg_dataset = PatchesDataset(qrcode_patches_dir=None,
                                    background_patches_dir=val_background_dir,
                                    device=device)
    val_same_qr_dataset = PatchesDataset(qrcode_patches_dir=same_qr_dir,
                                    background_patches_dir=None,
                                    device=device)
    val_same_bk_dataset = PatchesDataset(qrcode_patches_dir=None,
                                    background_patches_dir=same_bk_dir,
                                    device=device)

    val_qr_dataloader = DataLoader(val_qr_dataset, batch_size=10)
    val_bg_dataloader = DataLoader(val_bg_dataset, batch_size=10)
    val_same_qr_dataloader = DataLoader(val_same_qr_dataset, batch_size=10)
    val_same_bg_dataloader = DataLoader(val_same_bk_dataset, batch_size=10)

    with torch.no_grad():
        class_name = ['background', 'QRCode']
        #
        print("test qr code -------")
        for data in val_qr_dataloader:
            images, labels = data
            # images = torch.unsqueeze(images, dim=0)
            out_data = net(images)
            print(out_data)
            _, max_idx = torch.max(out_data.detach().cpu(), dim=1)
            print("class:", [class_name[idx] for idx in max_idx])
            break

        print("test background -------")
        for data in val_bg_dataloader:
            images, labels = data
            # images = torch.unsqueeze(images, dim=0)
            out_data = net(images)
            print(out_data)
            _, max_idx = torch.max(out_data.detach().cpu(), dim=1)
            print("class:", [class_name[idx] for idx in max_idx])
            break


        print("test same qr ----")
        for data in val_same_qr_dataloader:
            images, labels = data
            # images = torch.unsqueeze(images, dim=0)
            out_data = net(images)
            print(out_data)
            _, max_idx = torch.max(out_data.detach().cpu(), dim=1)
            print("class:", [class_name[idx] for idx in max_idx])
            break

        print("test same bg ----")
        for data in val_same_bg_dataloader:
            images, labels = data
            # images = torch.unsqueeze(images, dim=0)
            out_data = net(images)
            print(out_data)
            _, max_idx = torch.max(out_data.detach().cpu(), dim=1)
            print("class:", [class_name[idx] for idx in max_idx])
            break

