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

def train(net, dataloader, epoches, lr):

    cuda = torch.cuda.is_available()
    assert cuda  # 必須為 == True, cuda 失敗
    net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    # loss function
    mse = nn.MSELoss()

    CosineAnnealingWarmRestarts = True
    if CosineAnnealingWarmRestarts:
        alpha_ = epoches / 30  # hyp
        the_first_restart = int(epoches * (0.8 / alpha_))  # hyp
        T_mult = 2
        print(f"CosineAnnealingWarmRestarts: T_0:{the_first_restart}, T_mult:{T_mult}")
        train_scheduler = optim.lr_scheduler. \
            CosineAnnealingWarmRestarts(optimizer,
                                        T_0=the_first_restart,
                                        T_mult=T_mult, verbose=True)

    loss_recoder = []
    lr_recoder = []
    pbar = tqdm(range(epoches), smoothing=0.1, ncols=100)
    for epoch in pbar:
        avg_batch_loss = 0
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

        __loss = avg_batch_loss / batch_idx
        loss_recoder.append(__loss)
        pbar.set_description("Training progress: Epoch{0} loss={1:.6f}".format(epoch + 1, __loss))
    # loss
    fig = plt.figure("loss")
    plt.plot(loss_recoder)
    plt.yscale('log')
    fig.savefig("loss_recoder.png")
    # lr
    fig_lr = plt.figure("lr")
    plt.plot(lr_recoder)
    plt.yscale('log')
    fig_lr.savefig("lr_recoder.png")
    plt.clf()
    plt.cla()
    plt.close(fig_lr)
    plt.close(fig)
    return net

if __name__ == "__main__":

    my_dataset = my_dataset.QR_good_bad_dataset("./data", image_size=128)
    my_dataloader = DataLoader(my_dataset, batch_size=2, drop_last=False, pin_memory=False)
    net = my_model.Autoencoder()

    #
    net = train(net, my_dataloader, epoches=20, lr=1e-4)

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



