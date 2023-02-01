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
from tqdm import tqdm
import model as my_model
import dataloader as my_dataset
import torch.optim as optim
from datetime import datetime
from torch.utils.data import random_split
import util as myutil


# 時戳
current_time = lambda: datetime.now().strftime('%m%d_%H%M_%S')


def train(net, dataloader, val_dataloader, epoches, lr, save_root, SAVE_ROUND=1000):
    SAVE_ROOT = Path(save_root)
    myutil.ensure_dir(SAVE_ROOT)
    _stamp = current_time()
    LOSS_RECODER_TXT = SAVE_ROOT.joinpath(f"./loss_recoder_{_stamp}.txt")
    with open(LOSS_RECODER_TXT, mode='a', encoding='utf-8') as f:
        f.write(f"=========== Start train: {_stamp} ===========\n")
        f.write(f"train loss,\t valid loss,\t lr\n")


    cuda = torch.cuda.is_available()
    assert cuda  # 必須為 == True, cuda 失敗
    net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    # loss function
    mse = nn.MSELoss()
    # save 週期
    save_round = SAVE_ROUND
    save_cnt = 0
    #
    CosineAnnealingWarmRestarts = True
    if CosineAnnealingWarmRestarts:
        alpha_ = epoches / 30  # hyp
        the_first_restart = int(epoches * (0.8 / alpha_))  # hyp
        T_mult = 2
        print(f"CosineAnnealingWarmRestarts: T_0:{the_first_restart}, T_mult:{T_mult}")
        train_scheduler = optim.lr_scheduler. \
            CosineAnnealingWarmRestarts(optimizer,
                                        T_0=the_first_restart,
                                        T_mult=T_mult, verbose=False)

    # best saver
    save_best = myutil.SaveBestModel(SAVE_ROOT)

    loss_recoder = []
    loss_val_recoder = []
    lr_recoder = []
    pbar = tqdm(range(epoches), smoothing=0.1, ncols=100)
    for epoch in pbar:
        save_cnt += 1
        avg_batch_loss = 0
        avg_batch_val_loss = 0
        for batch_idx, data_good_bad in enumerate(dataloader):
            good, bad = data_good_bad
            good = good.cuda().float()/255.0
            bad = bad.cuda().float()/255.0
            bad = Variable(bad)
            optimizer.zero_grad()
            output = net(bad)
            loss = mse(output, good)
            avg_batch_loss += loss.detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            if CosineAnnealingWarmRestarts:
                train_scheduler.step(epoch + batch_idx / len(dataloader))
                lr_recoder.append(train_scheduler.get_last_lr()[0])
        # end of dataloader
        # calc train loss
        __loss = avg_batch_loss / batch_idx
        loss_recoder.append(__loss)

        # calc val loss
        net.eval()
        for batch_idx, data_good_bad in enumerate(val_dataloader):
            good, bad = data_good_bad
            good = good.cuda().float() / 255.0
            bad = bad.cuda().float() / 255.0
            bad = Variable(bad)
            optimizer.zero_grad()
            output = net(bad)
            loss = mse(output, good)
            avg_batch_val_loss += loss.detach().cpu().numpy()
        #
        __val_loss = avg_batch_val_loss / batch_idx
        save_best(__val_loss, net, epoch)
        loss_val_recoder.append(__val_loss)

        pbar.set_description("Training progress: Epoch{0} loss={1:.6f}, val_loss={2:.6f}".format(epoch + 1, __loss, __val_loss))
        # tranning save
        if save_cnt % save_round == 0:
            print("save model...")
            _pt_name = SAVE_ROOT.joinpath(f'./{current_time()}_e{epoch+1}.pt')
            torch.save(net.state_dict(), _pt_name)
            #  for colab special command
            #!cp /content/$_pt_name /content/gdrive/MyDrive/ColabNotebooks/$_pt_name
        # 紀錄 loss
        with open(LOSS_RECODER_TXT, mode='a', encoding='utf-8') as f:
            f.write(str(__loss) + ", " + str(__val_loss) + ", " + str(train_scheduler.get_last_lr()[0]) + "\n")
        # for colab special command
        # !cp -f /content/$LOSS_RECODER_TXT /content/gdrive/MyDrive/ColabNotebooks/$LOSS_RECODER_TXT

        #--------------------------------
        # loss
        plt.style.use('ggplot')
        fig = plt.figure("loss")
        plt.plot(loss_recoder, label='train loss')
        plt.plot(loss_val_recoder, label='valid loss')
        plt.legend()
        plt.yscale('log')
        fig.savefig(SAVE_ROOT.joinpath("loss_recoder.png"), dpi=300)
        # lr
        fig_lr = plt.figure("lr")
        plt.plot(lr_recoder)
        plt.yscale('log')
        fig_lr.savefig(SAVE_ROOT.joinpath("lr_recoder.png"), dpi=300)
        plt.clf()
        plt.cla()
        plt.close(fig_lr)
        plt.close(fig)
        plt.style.use('classic')
    return net


def make_test_view(net, dataloader, r, c, sub_size):
    from util import pad_2_square
    assert dataloader.batch_size == r * c

    net.cuda()
    with torch.no_grad():
        bad = None
        for batch_idx, data_good_bad in enumerate(dataloader):
            good, bad = data_good_bad
            rebuild = net(bad.cuda().float() / 255.0)
            break
        r_tmp = []
        # make bad
        bad = bad.numpy().squeeze()
        for j in range(r):
            c_tmp = []
            for i in range(c):
                idx = (j * c) + i
                c_tmp.append(pad_2_square(bad[idx], sub_size))
            r_tmp.append(np.hstack(c_tmp))
        res_bad = np.vstack(r_tmp)

        # make rebuild
        rebuild = rebuild.detach().cpu().numpy().squeeze() * 255.0
        r_tmp = []
        for j in range(r):
            c_tmp = []
            for i in range(c):
                idx = (j * c) + i
                c_tmp.append(pad_2_square(rebuild[idx], sub_size))
            r_tmp.append(np.hstack(c_tmp))
        res_rebuild = np.vstack(r_tmp)
    return np.hstack((res_bad, res_rebuild))
# def make_test_view(net, dataloader, r, c, sub_size, normalization=True):
#     from util import pad_2_square
#     assert dataloader.batch_size == r * c
#     net.eval()
#     net.cuda()
#     out = []
#     for batch_idx, data_good_bad in enumerate(dataloader):
#         good, bad = data_good_bad
#         origin_bad = np.array(bad.squeeze())
#         #good = good.cuda().float()/255.0
#         if normalization:
#             bad = bad.cuda().float() / 255.0
#         else:
#             bad = bad.cuda().float()
#         out = net(bad)
#         out = out.detach().cpu().numpy().squeeze()*255.0
#         break  # !!
#     r_tmp = []
#     # make bad
#     for j in range(r):
#         c_tmp = []
#         for i in range(c):
#             idx = (j * c) + i
#             c_tmp.append(pad_2_square(out[idx], sub_size))
#         r_tmp.append(np.hstack(c_tmp))
#     res_right = np.vstack(r_tmp)
#
#     # make  bad left
#     bad = np.array(origin_bad)
#     r_tmp = []
#     for j in range(r):
#         c_tmp = []
#         for i in range(c):
#             idx = (j * c) + i
#             c_tmp.append(pad_2_square(bad[idx], sub_size))
#         r_tmp.append(np.hstack(c_tmp))
#     res_left = np.vstack(r_tmp)
#     return np.hstack((res_left, res_right))


if __name__ == "__main__":

    good_bad_dataset = my_dataset.QR_good_bad_dataset("./data", image_size=128, image_channels=1)

    my_train_dataset, my_val_dataset, my_test_dataset = random_split(good_bad_dataset, [0.7, 0.2, 0.1],
                                                                     generator=torch.Generator().manual_seed(42))
    #my_dataloader, my_val_dataset, _ = random_split(my_dataset, [7, 3, 720])  # for fast test


    my_dataloader = DataLoader(my_train_dataset, batch_size=2, drop_last=False, pin_memory=False)
    val_dataloader = DataLoader(my_val_dataset, batch_size=2, drop_last=False, pin_memory=False)
    # view 用
    r, c = 5, 3
    assert r * c <= len(my_test_dataset)
    test_dataloader = DataLoader(my_test_dataset, batch_size=r*c, drop_last=False, pin_memory=False)

    torch.manual_seed(0)
    net = my_model.Autoencoder()

    #
    net = train(net, my_dataloader, my_val_dataset, save_root='1225_today', epoches=1, lr=1e-5, SAVE_ROUND=2)

    res = make_test_view(net, test_dataloader, r, c, sub_size=128)
    plt.imshow(res, cmap='gray');plt.show();


    # bad image
    test_batch = my_dataset[0][1].reshape(1, 1, 128, 128)
    # show bad batch
    _test_batch_show = np.array(test_batch.reshape(128, 128))
    plt.imshow(_test_batch_show, cmap=plt.cm.gray)
    fig1 = plt.gcf()
    fig1.savefig("before.png")


    output_batch = net(test_batch.cuda().float())
    # show output_batch batch
    _output_show = output_batch.reshape(128, 128).detach().cpu().numpy()
    plt.imshow(_output_show, cmap=plt.cm.gray)
    fig2 = plt.gcf()
    fig2.savefig("after.png")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Rebuild')
    #
    ax1.set_title('before')
    ax1.imshow(_test_batch_show)
    #
    ax2.set_title('after')
    ax2.imshow(_output_show)
    plt.tight_layout()
    plt.show()



