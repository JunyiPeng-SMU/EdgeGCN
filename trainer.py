# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import *
import os
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import preprocessing
from lifelines.statistics import logrank_test
from sklearn.metrics import brier_score_loss, roc_auc_score, recall_score,roc_curve, auc

def train_loss(data_loader):
    model.train()
    loss_all = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.label.to(device)
        loss = crit(output, label.long())
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(data_loader)


def test(loader):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.softmax(dim=1)
            predictions.append(pred.detach().cpu().numpy())
            labels.append(data.label.detach().cpu().numpy())
    predictions = np.hstack(predictions)
    labels = np.hstack(labels)
    auc = roc_auc_score(labels, predictions[:,1])
    acc = accuracy_score(labels, out.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
    return auc, acc

def test_loss(data_loader):
    model.eval()
    loss_all = 0
    # with torch.no_grad():
    for data in data_loader:
        data = data.to(device)
        output = model(data)
        label = data.label.to(device)
        loss = crit(output, label.long())
        loss_all += loss.item()
    return loss_all / len(data_loader)

if __name__ == "__main__":
    path = '/public/pengjunyi/Code/Node/OA/dataset/'
    crit = torch.nn.CrossEntropyLoss()
    dataset0 = OA_Dataset0(root=path, model='DESSIW', data_set=0)
    data_loader0 = DataLoader(dataset0, batch_size=len(dataset0))
    dataset1 = OA_Dataset0(root=path, model='DESSIW', data_set=1)
    data_loader1 = DataLoader(dataset1, batch_size=len(dataset1))
    num_fea = dataset0.num_features
    channel = 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EdgeCNN_Net(channel, num_fea,num_lay).to(device)
    model_name = 'EdgeGCN'
    item = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    for epoch in range(1, item):
        loss_train = train_loss(data_loader0)
        loss_test = test_loss(data_loader1)
        train_auc, train_acc = test(data_loader0)
        test_auc, test_acc = test(data_loader1)
        print('Epoch: {:03d}, Loss_train: {:.5f}, Loss_val: {:.5f},Train auc: {:.5f}, Test auc: {:.5f}'.format(epoch, loss_train, loss_test, train_auc, test_auc))
    model_path = os.path.join("/public/pengjunyi/Code/Node/OA/model/")
    torch.save(model.state_dict(), os.path.join(model_path, 'model_EdgeGCN.pth'))