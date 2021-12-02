import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')



class QRCode_CNN(nn.Module):
    def __init__(self, act_func=F.relu, drop=0.0):
        super(QRCode_CNN, self).__init__()
        self.drop = drop
        self.drop_layer = nn.Dropout2d(p=self.drop)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)  #
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        # fc1: 5*5 是要根據前剛的網路計算出來的
        # fc1: 根據論文是使用 300 個 neurons.
        self.fc1 = nn.Linear(12 * 5 * 5, 300)
        # fc1: 根據論文是使用 2 個 neurons，直接皆輸出.
        self.fc2 = nn.Linear(300, 2)
        self.act = act_func

    def forward(self, x):
        x = x.view((-1, 1, 32, 32))
        x = self.act(self.conv1(x))
        x = self.drop_layer(self.pool(x))
        x = self.act(self.conv2(x))
        x = self.drop_layer(self.pool(x))
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        # !!! pytorch 做分類問題時不用自己套 softmax，
        # 他會在 nn.CrossEntropyLoss 預設套用 LogSoftmax+NLLLoss。
        #
        return x


if __name__ == "__main__":
    net = QRCode_CNN()
    mini_batch = torch.randn((64, 1, 32, 32))
    pred_y = net(mini_batch)
    print(pred_y)
    print(pred_y.shape)
