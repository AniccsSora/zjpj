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
    if weight is not None:
        print(f"[qr, bg]: {weight[1]}, {weight[0]}")
        weight = torch.tensor([weight[1], weight[0]]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for epoch in range(epochs):
        avg_loss_in_a_epoch, cnt = 0, 0
        for data in dataloader:
            train_images, train_labels = data
            y_pred = net(train_images)

            loss = criterion(y_pred.to(device), train_labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ---
            avg_loss_in_a_epoch += loss.item()
            cnt += 1

        print(f'epoch: {epoch + 1}, loss: {avg_loss_in_a_epoch/cnt}')
    return net


if __name__ == "__main__":
    patches_dataset = PatchesDataset(qrcode_patches_dir='./data/pathes_of_qrcode_32x32',
                                     background_patches_dir='./data/background_patch',
                                     device=device)
    qr_bg_ratio = torch.tensor(patches_dataset.weight)  # qrcode 與 背景 資料比例。

    train_dataloader = DataLoader(patches_dataset, batch_size=16, shuffle=True)

    # test
    tdata = next(iter(train_dataloader))
    images, labels = tdata

    net = QRCode_CNN()

    net = train(train_dataloader, net, lr=1e-4, epochs=1, weight=qr_bg_ratio)

    val_background_dir="./data/val_patch_False"
    val_qrcode_dir = "./data/val_patch_True"

    val_qr_dataset = PatchesDataset(qrcode_patches_dir=val_qrcode_dir,
                                     background_patches_dir=None,
                                     device=device)
    val_bg_dataset = PatchesDataset(qrcode_patches_dir=None,
                                    background_patches_dir=val_background_dir,
                                    device=device)

    val_qr_dataloader = DataLoader(val_qr_dataset, batch_size=1)
    val_bg_dataloader = DataLoader(val_bg_dataset, batch_size=1)

    with torch.no_grad():
        #
        print("test qr code -------")
        for data in val_qr_dataloader:
            images, labels = data
            # images = torch.unsqueeze(images, dim=0)
            out_data = net(images)
            print(out_data)

        print("test background -------")
        for data in val_bg_dataloader:
            images, labels = data
            # images = torch.unsqueeze(images, dim=0)
            out_data = net(images)
            print(out_data)

