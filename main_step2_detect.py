import model
import torch
import torch.nn as nn
from dataloader.PatchesDataset import get_Dataloader
from torch.nn.functional import one_hot

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_path = "./log_save/20211206_2247_31/weight.pt"
    qr_dir = "./data/val_patch_True"
    bg_dir = "./data/val_patch_False"
    val_dataloader = get_Dataloader(qrcode_dir=qr_dir, background_dir=bg_dir)
    net = model.QRCode_CNN()
    net.load_state_dict(torch.load(weight_path))
    net.cuda()
    with torch.no_grad():
        loss_criterion = nn.CrossEntropyLoss()
        total_loss, cnt = 0, 0
        for data in val_dataloader:
            images, labels = data
            # 因為是練好的 model 預期出來需要是機率
            # 並且要跟 one-hot 算 loss，(分布與分布之間的 loss)
            pred_y = net(images).softmax(dim=1)
            labels = one_hot(labels.long(), num_classes=2)
            total_loss += loss_criterion(labels.float(), pred_y)
            cnt += 1
        avg_loss = total_loss/cnt
        print("val_loss:", avg_loss.detach().cpu().numpy())
