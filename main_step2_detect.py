import model
import torch
from dataloader.PatchesDataset import get_Dataloader

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_path = "./log_save/20211206_2247_31/weight.pt"
    qr_dir = "./data/val_patch_True"
    bg_dir = "./data/val_patch_False"
    val_dataloader = get_Dataloader(qrcode_dir=qr_dir, background_dir=bg_dir)
    net = model.QRCode_CNN()
    net.load_state_dict(torch.load(weight_path))
    net.eval()
    net.cuda()
    for data in val_dataloader:
        images, labels = data
        print(net(images))
