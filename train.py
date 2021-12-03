import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.PatchesDataset import PatchesDataset
from torch.utils.data import DataLoader
from model import QRCode_CNN
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def param_foolproof_1(draw, val_dataloader):
    if val_dataloader is None:
        assert draw is None
    if draw is None:
        assert val_dataloader is None


def train(dataloader, net, lr, epochs, weight=None,
          class_name=['background', 'QRCode'],
          draw=None,
          val_dataloader=None):
    # 參數防呆
    param_foolproof_1(draw, val_dataloader)

    if torch.cuda.is_available():
        net.cuda()
    if weight is not None:
        print("Dataset weight:", weight)
        weight = torch.tensor([weight['QRCode'], weight['background']]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    eval_loss = {
        "train": [],
        "val": []
    }
    for epoch in range(epochs):
        avg_loss_in_a_epoch, cnt = 0, 0
        for data in dataloader:
            train_images, train_labels = data
            optimizer.zero_grad()
            y_pred = net(train_images)
            loss = criterion(y_pred.to(device), train_labels.to(device))
            loss.backward()
            optimizer.step()
            # ---
            avg_loss_in_a_epoch += loss.item()
            cnt += 1
        if val_dataloader is not None:
            with torch.no_grad():
                val_total_loss = 0.0
                val_cnt = 0
                for val_data in val_dataloader:
                    val_images, val_labels = val_data
                    val_y_pred = net(val_images)
                    val_total_loss += criterion(val_y_pred.to(device), val_labels.to(device))
                    val_cnt += 1
                _ = val_total_loss / val_cnt
                eval_loss['val'].append(_)
        # end of validation.
        train_loss = avg_loss_in_a_epoch/cnt
        eval_loss['train'].append(train_loss)
        print(f'epoch: {epoch + 1}, loss: {train_loss}')
    return net, eval_loss


if __name__ == "__main__":
    patches_dataset = PatchesDataset(qrcode_patches_dir='./data/pathes_of_qrcode_32x32',
                                     background_patches_dir='./data/background_patch',
                                     device=device)

    train_dataloader = DataLoader(patches_dataset, batch_size=64, shuffle=True)

    # test
    tdata = next(iter(train_dataloader))
    images, labels = tdata

    net = QRCode_CNN(drop=0.1)

    net = train(train_dataloader, net, lr=1e-3, epochs=2, weight=patches_dataset.weight)
    torch.save(net.state_dict(), './trained.pt')
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

