#reference: https://blog.csdn.net/mylove0414/article/details/55805974
#reference:https://github.com/louisenaud/stock_prediction/blob/master/src/run.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset 
import tqdm
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser("lstm")
parser.add_argument("-data_path", type=str, default="./data/stock_dataset/dataset_1.csv", help="dataset path")

args = parser.parse_args()

class LSTM_model(nn.Module):
    def __init__(self, input_dim, rnn_unit):
        super(LSTM_model, self).__init__()
        self.dim = input_dim
        self.rnn_unit = rnn_unit
        self.emb_layer = nn.Linear(input_dim, rnn_unit)
        self.out_layer = nn.Linear(rnn_unit, input_dim)
        self.rnn_layer = 2
        self.lstm = nn.LSTM(input_size=rnn_unit, hidden_size=rnn_unit, num_layers=self.rnn_layer, batch_first=True)
    
    def init_hidden(self, x):
        batch_size = x.shape[0]
        rtn = (torch.zeros(self.rnn_layer, batch_size, self.rnn_unit, device=x.device).requires_grad_(),
                torch.zeros(self.rnn_layer, batch_size, self.rnn_unit, device=x.device).requires_grad_())
        return rtn

    def forward(self, input_data, h0=None):
        # batch x time x dim
        h0 = h0 if h0 else self.init_hidden(input_data)
        x = self.emb_layer(input_data)
        
        output, hidden = self.lstm(x, h0)
        
        out = self.out_layer(output[:,-1,:].squeeze()).squeeze()
        return out, hidden



class StockDataset(Dataset):
    def __init__(self, file_path, T=10, train_flag=True):
        # read data
        with open(file_path, "r", encoding="GB2312") as fp:
            data_pd = pd.read_csv(fp)
        self.train_flag = train_flag
        self.data_train_ratio = 0.8
        self.T = T # use 10 data to pred
        if train_flag:
            self.data_len = int(self.data_train_ratio * len(data_pd))
            data_all = np.array(data_pd['最高价'])[::-1]
            data_all = (data_all-np.mean(data_all))/np.std(data_all)
            self.data = data_all[:self.data_len]
        else:
            self.data_len = int((1-self.data_train_ratio) * len(data_pd))
            data_all = np.array(data_pd['最高价'])[::-1]
            data_all = (data_all-np.mean(data_all))/np.std(data_all)
            self.data = data_all[-self.data_len:]
        print("data len:{}".format(self.data_len))


    def __len__(self):
        return self.data_len-self.T

    def __getitem__(self, idx):
        return self.data[idx:idx+self.T], self.data[idx+self.T]


def l2_loss(pred, label):
    loss = torch.nn.functional.mse_loss(pred, label, size_average=True)
    return loss

def train_once(model, dataloader, optimizer):
    model.train()
    
    loader = tqdm.tqdm(dataloader)
    loss_epoch = 0
    for idx, (data, label) in enumerate(loader):
        # data: batch, time
        data = data.unsqueeze(2)
        data, label = Variable(data.float()), Variable(label.float())
        output, _ = model(data)
        optimizer.zero_grad()
        loss = l2_loss(output, label)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.detach().item()

    loss_epoch /= len(loader)
    return loss_epoch
    #print("epoch:{:5d}, loss:{:6.3f}".format(epoch, loss_epoch))


def eval_once(model, dataloader):
    model.eval()
    loader = tqdm.tqdm(dataloader)
    loss_epoch = 0
    for idx, (data, label) in enumerate(loader):
        # data: batch, time x 1
        data = data.unsqueeze(2)
        data, label = data.float(), label.float()
        output, _ = model(data)
        loss = l2_loss(output, label)
        loss_epoch += loss.detach().item()
    loss_epoch /= len(loader)
    return loss_epoch

def eval_plot(model, dataloader):
    dataloader.shuffle = False
    preds = []
    labels = []
    model.eval()
    loader = tqdm.tqdm(dataloader)
    for idx, (data, label) in enumerate(loader):
        # data: batch, time x 1
        data, label = data.float().unsqueeze(2), label.float()
        output, _ = model(data)
        preds+=(output.detach().tolist())
        labels+=(label.detach().tolist())

    #plot
    fig, ax = plt.subplots()
    data_x = list(range(len(preds)))
    ax.plot(data_x, preds, **{"color":"blue", "linestyle":"-.",  "marker":","})
    ax.plot(data_x, labels, **{"color":"red", "linestyle":":", "marker":","})
    plt.show()


def main():
    dataset_train = StockDataset(file_path=args.data_path)
    dataset_val = StockDataset(file_path=args.data_path, train_flag=False)
    
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=64, shuffle=True)

    model = LSTM_model(input_dim=1, rnn_unit=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    total_epoch = 100
    #eval_plot(model, val_loader)
    for epoch_idx in range(total_epoch):
        train_loss = train_once(model, train_loader, optimizer)
        print("stage: train, epoch:{:5d}, loss:{:6.3f}".format(epoch_idx, train_loss))
        if epoch_idx%10==0:
            eval_loss = eval_once(model, val_loader)
            print("stage: test, epoch:{:5d}, loss:{:6.3f}".format(epoch_idx, eval_loss))
    eval_plot(model, val_loader)





if __name__ == "__main__":
    main()


