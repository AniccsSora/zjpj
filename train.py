import torch
import torch.nn as nn
import torch.optim as optim

def train(dataloader, net, lr, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    # for epoch in range(epochs):
    #     y_pred = model(X_train)
    #     loss = criterion(y_pred, Y_train)
    #     print('epoch: ', epoch + 1, ' loss: ', loss.item())
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

if __name__== "__main__":
    pass