from unet import UNet
import config as cfg
import torch


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('device: ', device)
    unet = UNet(n_classes=1, depth=cfg.depth, padding=True).to(device)
