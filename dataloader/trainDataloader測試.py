from PatchesDataset import PatchesDataset
import torch
from torch.utils.data import DataLoader


if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patches_dataset = PatchesDataset(qrcode_patches_dir='../data/pathes_of_qrcode_32x32',
                                     background_patches_dir='../data/background_patch',
                                     device=device)

    train_dataloader = DataLoader(patches_dataset, batch_size=128, shuffle=True)

    train_images, train_labels = next(iter(train_dataloader))

    print(train_images, train_labels)
    #print("device:", device)