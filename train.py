import torch
import torch.optim as optim
import os
from fudanDataSet import fudanDataSet, fudanDataSetTest
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class model(nn.Module):
    def __init__(self):
        super().__init__()
        ## 对文本长度进行降维
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=200,
                      out_channels=200, kernel_size=5),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5)
        )

        ## LSTM+全连接
        self.lstm = nn.LSTM(200, 64, 3, batch_first=True)
        self.fc1 = nn.Linear(64, 20)

    def forward(self, x):
        x = self.conv(x).to(device)
        x = x.permute(0, 2, 1).to(device)
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.fc1(r_out[:, -1, :])
        return out


def train():
    net = model().to(device)
    net.load_state_dict(torch.load(f'model/net{81}.pth'))
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    net.train()
    dataset = fudanDataSet()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(200):
        for idx, (data, mark) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            z = net(data).to(device)
            list1 = torch.zeros_like(z).to(device)
            for i in range(z.shape[0]):
                list1[i][int(mark[i]) - 1] = 1
            # print(mark)
            # print(z)
            # print(list1)
            # print(z - list1)
            # print(torch.square((z - list1)))
            loss = torch.sum(torch.square(z - list1)).to(device)
            print(f'loss:{loss} epoch:{epoch}')
            loss.backward()
            optimizer.step()
            torch.save(net.state_dict(), f'model/net{epoch}.pth')


train()

def test():
    net = model().to(device)
    net.eval()
    dataset = fudanDataSetTest()
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    data_sum = len(dataset)
    with open('result2', 'w') as resultfile:
        for index in range(1):
            right_sum = 0
            print(f"model{index}")
            resultfile.write(f"model{index}")
            net.load_state_dict(torch.load(f'model/model/net{index}.pth', map_location=device))
            for idx, (data, mark) in enumerate(dataloader):
                batch_sum = 0
                data = data.to(device)
                z = net(data).to(device)
                list1 = torch.zeros_like(z).to(device)
                for i in range(z.shape[0]):
                    list1[i][int(mark[i]) - 1] = 1
                _, z = torch.max(z, dim=1)
                for i in range(z.shape[0]):
                    if z[i] == int(mark[i]) - 1:
                        right_sum += 1
                        batch_sum += 1
                print(f"该批样本:{z.shape[0]}, 正确分类:{batch_sum}, 准确率:{batch_sum / z.shape[0]}")
            print(f"model{index} 样本:{data_sum}, 正确分类:{right_sum}, 准确率:{right_sum / data_sum}")
            resultfile.write(f"model{index} 样本:{data_sum}, 正确分类:{right_sum}, 准确率:{right_sum / data_sum}")




# test()
