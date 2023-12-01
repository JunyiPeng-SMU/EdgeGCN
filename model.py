#-*- coding:utf-8 -*-
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from torch_geometric.nn.models import EdgeCNN
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import random
import numpy as np
import os
class OA_Dataset0(InMemoryDataset):
    def __init__(self, root,  model, data_set, transform=None, pre_transform=None):
        self.data_set = data_set
        self.model = model
        if self.model == 'IW':
            # self.fealist = fea_peng_IW
            fea = pd.read_excel("/public/pengjunyi/Code/Node/OA/mrmr_IW96.xlsx", sheet_name='Sheet1')
            self.feature_list = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14',
                            'r15', 'r16', 'r17', 'r18', 'r19', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25', 'r26', 'r27',
                            'r28', 'r29', 'r30', 'r31', 'r32', 'r33', 'r34', 'r35', 'r36', 'r37', 'r38', 'r39', 'r40',
                            'r41', 'r42', 'r43', 'r44', 'r45', 'r46', 'r47', 'r48', 'r49', 'r50', 'r51', 'r52', 'r53',
                            'r54', 'r55', 'r56', 'r57', 'r58', 'r59', 'r60', 'r61', 'r62', 'r63', 'r64', 'r65', 'r66',
                            'r67', 'r68', 'r69', 'r70', 'r71', 'r72', 'r73', 'r74', 'r75', 'r76', 'r77', 'r78', 'r79',
                            'r80', 'r81', 'r82', 'r83', 'r84', 'r85', 'r86', 'r87', 'r88', 'r89', 'r90', 'r91', 'r92',
                            'r93', 'r94', 'r95', 'r96']
            if data_set==0:
                self.df = fea[fea['split'] == 0]
                self.adjacent = np.load('/public/pengjunyi/Code/Node/OA/npy_data/npy_8/fold0_train.npy', allow_pickle=True).item()
            elif data_set == 1:
                self.df = fea[fea['split'] == 1]
                self.adjacent = np.load('/public/pengjunyi/Code/Node/OA/npy_data/npy_8/fold0_test.npy',allow_pickle=True).item()
        elif self.model == 'DESS':
            # self.fealist = fea_peng_DESS
            fea = pd.read_excel("/public/pengjunyi/Code/Node/OA/mrmr_DESS96.xlsx", sheet_name='Sheet1')
            self.feature_list = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14',
                            'r15', 'r16', 'r17', 'r18', 'r19', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25', 'r26', 'r27',
                            'r28', 'r29', 'r30', 'r31', 'r32', 'r33', 'r34', 'r35', 'r36', 'r37', 'r38', 'r39', 'r40',
                            'r41', 'r42', 'r43', 'r44', 'r45', 'r46', 'r47', 'r48', 'r49', 'r50', 'r51', 'r52', 'r53',
                            'r54', 'r55', 'r56', 'r57', 'r58', 'r59', 'r60', 'r61', 'r62', 'r63', 'r64', 'r65', 'r66',
                            'r67', 'r68', 'r69', 'r70', 'r71', 'r72', 'r73', 'r74', 'r75', 'r76', 'r77', 'r78', 'r79',
                            'r80', 'r81', 'r82', 'r83', 'r84', 'r85', 'r86', 'r87', 'r88', 'r89', 'r90', 'r91', 'r92',
                            'r93', 'r94', 'r95', 'r96']
            if data_set == 0:
                self.df = fea[fea['split'] == 0]
                self.adjacent = np.load('/public/pengjunyi/Code/Node/OA/npy_data/npy_8/fold0_train.npy', allow_pickle=True).item()
            elif data_set == 1:
                self.df = fea[fea['split'] == 1]
                self.adjacent = np.load('/public/pengjunyi/Code/Node/OA/npy_data/npy_8/fold0_test.npy', allow_pickle=True).item()
        elif self.model == 'DESSIW':
            # self.fealist = fea_peng_DESS
            fea = pd.read_excel("/public/pengjunyi/Code/Node/OA/mrmr_DESSIW96.xlsx", sheet_name='Sheet1')
            self.feature_list = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13','r14',
                                 'r15', 'r16', 'r17', 'r18', 'r19', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25', 'r26','r27',
                                 'r28', 'r29', 'r30', 'r31', 'r32', 'r33', 'r34', 'r35', 'r36', 'r37', 'r38', 'r39','r40',
                                 'r41', 'r42', 'r43', 'r44', 'r45', 'r46', 'r47', 'r48', 'r49', 'r50', 'r51', 'r52','r53',
                                 'r54', 'r55', 'r56', 'r57', 'r58', 'r59', 'r60', 'r61', 'r62', 'r63', 'r64', 'r65','r66',
                                 'r67', 'r68', 'r69', 'r70', 'r71', 'r72', 'r73', 'r74', 'r75', 'r76', 'r77', 'r78','r79',
                                 'r80', 'r81', 'r82', 'r83', 'r84', 'r85', 'r86', 'r87', 'r88', 'r89', 'r90', 'r91','r92',
                                 'r93', 'r94', 'r95', 'r96', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9','w10', 'w11', 'w12', 'w13', 'w14',
                                 'w15', 'w16', 'w17', 'w18', 'w19', 'w20', 'w21', 'w22', 'w23', 'w24', 'w25', 'w26','w27',
                                 'w28', 'w29', 'w30', 'w31', 'w32', 'w33', 'w34', 'w35', 'w36', 'w37', 'w38', 'w39',
                                 'w40','w41', 'w42', 'w43', 'w44', 'w45', 'w46', 'w47', 'w48', 'w49', 'w50', 'w51', 'w52',
                                 'w53','w54', 'w55', 'w56', 'w57', 'w58', 'w59', 'w60', 'w61', 'w62', 'w63', 'w64', 'w65',
                                 'w66','w67', 'w68', 'w69', 'w70', 'w71', 'w72', 'w73', 'w74', 'w75', 'w76', 'w77', 'w78',
                                 'w79','w80', 'w81', 'w82', 'w83', 'w84', 'w85', 'w86', 'w87', 'w88', 'w89', 'w90', 'w91',
                                 'w92','w93', 'w94', 'w95', 'w96']
            if data_set == 0:
                self.df = fea[fea['split'] == 0]
                self.adjacent = np.load('/public/pengjunyi/Code/Node/OA/npy_data/npy_8/fold0_train.npy', allow_pickle=True).item()
            elif data_set == 1:
                self.df = fea[fea['split'] == 1]
                self.adjacent = np.load('/public/pengjunyi/Code/Node/OA/npy_data/npy_8/fold0_test.npy', allow_pickle=True).item()
        super(OA_Dataset0, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        if self.data_set == 0 and self.model == 'IW':
            return ['/public/pengjunyi/Code/Node/OA/dataset_mrmr/matrix8/OA_IW_train.dataset']
        if self.data_set == 1 and self.model == 'IW':
            return ['/public/pengjunyi/Code/Node/OA/dataset_mrmr/matrix8/OA_IW_test.dataset']
        if self.data_set == 0 and self.model == 'DESS':
            return ['/public/pengjunyi/Code/Node/OA/dataset_mrmr/matrix8/OA_DESS_train.dataset']
        if self.data_set == 1 and self.model == 'DESS':
            return ['/public/pengjunyi/Code/Node/OA/dataset_mrmr/matrix8/OA_DESS_test.dataset']
        if self.data_set == 0 and self.model == 'DESSIW':
            return ['/public/pengjunyi/Code/Node/OA/dataset_mrmr/matrix8/OA_DESSIW_train.dataset']
        if self.data_set == 1 and self.model == 'DESSIW':
            return ['/public/pengjunyi/Code/Node/OA/dataset_mrmr/matrix8/OA_DESSIW_test.dataset']
    def download(self):
        pass
    def process(self):
        data_list = []
        # process by session_id
        grouped = self.df.groupby('session_id_fold0')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id_fold0)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id_fold0 == session_id, self.feature_list].values
            node_features = torch.tensor(node_features, dtype=torch.float)
            b = np.array(self.adjacent['list_a']).astype(dtype=int).tolist()
            d = np.array(self.adjacent['list_b']).astype(dtype=int).tolist()
            va = np.array(self.adjacent['values']).astype(dtype=int).tolist()
            edge_index = torch.tensor([b, d], dtype=torch.long)
            edge_attr = torch.tensor(va, dtype=torch.float)
            x = node_features
            label1 = torch.tensor(group.LABEL1.values, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, label=label1)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class EdgeCNN_Net(torch.nn.Module):
    def __init__(self,channel,num_features,num_layer):
        super(EdgeCNN_Net, self).__init__()
        self.conv01 = EdgeCNN(num_features, channel, num_layer,2,dropout=0.5)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv01(x=x,edge_index=edge_index)
        return x