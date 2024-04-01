import pandas as pd
import torch
from torch.utils.data import Dataset
import random as rd
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import torch.nn as nn
import torch.nn.functional as F

class scRNADataset(Dataset):
    def __init__(self,train_set,num_gene,flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag


    def __getitem__(self, idx):
        train_data = self.train_set[:,:2]
        train_label = self.train_set[:,-1]

        if self.flag:
            # train_len = len(train_label)
            # train_tan = np.zeros([train_len,2])
            # train_tan[:,0] = 1 - train_label
            # train_tan[:,1] = train_label
            # train_label = train_tan
            data = train_data[idx].astype(np.int64)
            label = train_label[idx].astype(np.int64)
        else:
            data = train_data[idx].astype(np.int64)
            label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.train_set)


    def Adj_Generate(self,TF_set,direction=False, loop=False):

        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)


        for pos in self.train_set:

            tf = pos[0]
            target = pos[1]

            if direction == False:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0


        if loop:
            adj = adj + sp.identity(self.num_gene)

        adj = adj.todok()
        return adj




    def Covariance_Generate(self, x_data,tf_data,target_data,direction=False, loop=False):

        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)
        ## 测试
        tf = tf_data.cpu().numpy()
        print(tf)

        ##
        for pos_tf in tf:
            r_list = []
            #r_index = []
            print("{}/{}".format(pos_tf,len(target_data)))
            count = 0
            for pos_target in range(len(target_data)):
                ##
                cos_s = F.cosine_similarity(x_data[pos_tf], x_data[pos_target], dim=0)
                cos_sim = cos_s.abs().cpu().numpy()
                r_list.append(cos_s)

                if cos_sim > 0.1:
                    count = count + 1
                    adj[pos_tf, pos_target] = 1.0
                    ##
                    adj[pos_target, pos_tf] = 1.0
            print("大于阈值的有：{}".format(count))
            # a_with_index = [(val, idx) for idx, val in enumerate(r_list)]
            # #
            # sorted_a = sorted(a_with_index, key=lambda x: x[0], reverse=True)[:100]
            # print("a_with_index:".format(a_with_index))
            # print(sorted_a)


                # #
                # r_index = [x[1] for x in sorted_a]
                # if direction == False:
                #     for j in r_index:
                #         adj[pos_tf, j] = 1.0

                ##
                # adj[pos_tf, pos_target] = cos_sim
                # adj[pos_target, pos_tf] = cos_sim

            #print("当前tf与所有target的相似度：{}".format(r_list))

            ##
            #
            # a_with_index = [(val, idx) for idx, val in enumerate(r_list)]
            # #
            # sorted_a = sorted(a_with_index, key=lambda x: x[0], reverse=True)[:100]
            # #
            # r_index = [x[1] for x in sorted_a]
            # if direction == False:
            #     for j in r_index:
            #         adj[pos_tf, j] = 1.0

        # for pos_tf in tf_data:
        #     r_list = []
        #     r_index= []
        #     for pos_target in target_data:
        #         ##
        #         h1 = x_data[pos_tf]
        #         h2 = x_data[pos_target]
        #         cos_r = pass
        #         r_list.append(cos_r)
        #
        #     ##
        #
        #     #
        #     a_with_index = [(val, idx) for idx, val in enumerate(r_list)]
        #     #
        #     sorted_a = sorted(a_with_index, key=lambda x: x[0], reverse=True)[:100]
        #     #
        #     r_index = [x[1] for x in sorted_a]
        #
        #     if direction == False:
        #         for j in r_index:
        #             adj[pos_tf, j] = 1.0


        # if loop:
        #     adj = adj + sp.identity(self.num_gene)

        adj = adj.todok()

        return adj




class load_data():
    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self,data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)

        return epr.T


    def exp_data(self):
        data_feature = self.data.values

        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)

        return data_feature


def adj2saprse_tensor(adj):
    coo = adj.tocoo()
    i = torch.LongTensor(np.array([coo.row, coo.col])) ##
    v = torch.from_numpy(coo.data).float()

    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
    return adj_sp_tensor





def Evaluation(y_true, y_pred,flag=False):
    if flag:
        # y_p = torch.argmax(y_pred,dim=1)
        y_p = y_pred[:,-1]
        y_p = y_p.cpu().detach().numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.cpu().detach().numpy()
        y_p = y_p.flatten()


    y_t = y_true.cpu().numpy().flatten().astype(int)

    AUC = roc_auc_score(y_true=y_t, y_score=y_p)


    AUPR = average_precision_score(y_true=y_t,y_score=y_p)
    AUPR_norm = AUPR/np.mean(y_t)


    return AUC, AUPR, AUPR_norm


def causal_evaluation(y_true,y_pred):

    ##
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().numpy().flatten().astype(int)
    print(y_pred)
    print(y_true)
    count = 0
    for i in range(len(y_pred)):
        max_index = y_pred[i].argmax()
        if max_index == y_true[i]:
            count = count + 1
    acc = count/len(y_true)

    return acc




def normalize(expression):
    std = StandardScaler()
    epr = std.fit_transform(expression)

    return epr



def Network_Statistic(data_type,net_scale,net_type):

    if net_type =='STRING':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale



    elif net_type == 'Non-Specific':

        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Specific':
        dic = {'hESC500': 0.164, 'hESC1000': 0.165,'hHEP500': 0.379, 'hHEP1000': 0.377,'mDC500': 0.085,
               'mDC1000': 0.082,'mESC500': 0.345, 'mESC1000': 0.347,'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565,'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Lofgof':
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}

        query = 'mESC' + str(net_scale)
        scale = dic[query]
        return scale

    else:
        raise ValueError































