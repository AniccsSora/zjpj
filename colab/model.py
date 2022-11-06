import torch
from torch import nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.in_channels = 1

        self.conv1 = nn.Conv2d(self.in_channels, 8, kernel_size=5, stride=5, padding=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        # self.encoder = nn.Sequential(
        #     F.relu(nn.Conv2d(self.in_channels, 8, kernel_size=3, padding=1)),
        #     #nn.MaxPool2d(2, stride=2),
        #     F.relu(nn.Conv2d(8, 16, kernel_size=3, padding=2)),
        #     #nn.MaxPool2d(2, stride=2),
        #     F.relu(nn.Conv2d(16, 32, kernel_size=3, padding=3)),
        #     #nn.MaxPool2d(2, stride=2),
        #     )
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(32, 16, kernel_size=3, padding=0),
        #     nn.ReLU(True),
        #     #nn.Upsample(scale_factor=2),
        #     nn.Conv2d(16, 8, kernel_size=3, padding=0),
        #     nn.ReLU(True),
        #     #nn.Upsample(scale_factor=2),
        #     nn.Conv2d(8, self.in_channels, kernel_size=3, padding=0),
        #     nn.ReLU(True),
        #     #nn.Upsample(scale_factor=2),
        #     nn.ReLU(True)
        #     )
        self.conv1T = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=0)
        self.conv2T = nn.ConvTranspose2d(16, 8, kernel_size=3, padding=0)
        self.conv3T = nn.ConvTranspose2d(8, self.in_channels, kernel_size=5, stride=5, padding=1)


    def forward(self, x):
        #print("in :", x.shape)
        # encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #print("latent:", x.shape)
        # decoder
        x = F.relu(self.conv1T(x))
        x = F.relu(self.conv2T(x))
        x = F.relu(self.conv3T(x))
        #print("out:", x.shape)
        return x


if __name__ == "__main__":


    fake_batch = torch.randn(2, 1, 128, 128).cuda()
    AE = Autoencoder().cuda()
    output = AE(fake_batch)
    output = output.detach().cpu().numpy()


    print(fake_batch.shape)
    print(output.shape)