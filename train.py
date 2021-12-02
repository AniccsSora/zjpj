import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.PatchesDataset import PatchesDataset
from torch.utils.data import DataLoader
from model import QRCode_CNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(dataloader, net, lr, epochs, weight=None):
    if torch.cuda.is_available():
        net.cuda()

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for epoch in range(epochs):
        avg_loss_in_a_epoch, cnt = 0, 0
        for data in dataloader:
            train_images, train_labels = data
            y_pred = net(train_images)
            loss = criterion(y_pred, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ---
            avg_loss_in_a_epoch += loss.item()
            cnt += 1

        print(f'epoch: {epoch + 1}, loss: {avg_loss_in_a_epoch/cnt}')

if __name__== "__main__":
    patches_dataset = PatchesDataset(qrcode_patches_dir='./data/pathes_of_qrcode_32x32',
                                     background_patches_dir='./data/background_patch',
                                     device=device)
    qr_bg_ratio = torch.tensor(patches_dataset.weight)  # qrcode 與 背景 資料比例。

    train_dataloader = DataLoader(patches_dataset, batch_size=128, shuffle=True)

    # test
    tdata = next(iter(train_dataloader))
    images, labels = tdata

    net = QRCode_CNN()

    train(train_dataloader, net, lr=1e-5, epochs=10, weight=qr_bg_ratio)