from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from gTrans import GTransformer
from torch.optim.lr_scheduler import StepLR
#import scipy.sparse as sp
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  causal_evaluation
import pandas as pd
#from torch.utils.tensorboard import SummaryWriter
#from Code.PytorchTools import EarlyStopping
import numpy as np
import random
#import glob
import os
from sklearn.decomposition import PCA
from sklearn import metrics

#import time
import argparse
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
time_point = 3
#d_type = 'mDC'
d_type = 'mHSC-E'
#d_type_2 = 'sim2'
d_short = 'mEc3_expression'
#d_short = 'mDc3_expression'
#d_short = 'h2_expression'
model_version = 'v7_causal_test_'
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default= 15, help='Number of epoch.')
#parser.add_argument('--epochs', type=int, default= 5, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3,3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=512, help='The size of each batch') ##
#parser.add_argument('--batch_size', type=int, default=8, help='The size of each batch')  ##
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type',type=str,default='dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')
#parser.add_argument('--time_point', type=int, default= 4, help='Number of epoch.')
args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


"""
Non-Specific
mHSC-L learning rate = 3e-5
"""
data_type = 'hESC'
num = 500


def embed2file(tf_embed,tg_embed,gene_file,tf_path,target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file)

    tf_embed = pd.DataFrame(tf_embed,index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)


net_type = "Specific"
## 返回网络密度target/tf*gene
# density = Network_Statistic(data_type,num,net_type)
# exp_file = './'+data_type+'/TFs+'+str(num)+'/BL--ExpressionData.csv'
# tf_file = './'+data_type+'/TFs+'+str(num)+'/TF.csv'
# target_file = './'+data_type+'/TFs+'+str(num)+'/Target.csv'

density = 1/36
# exp_file1 = './raw_data/mesc1/m1_expression0.csv'
# exp_file2 = './raw_data/mesc1/m1_expression1.csv'
# exp_file3 = './raw_data/mesc1/m1_expression2.csv'
# exp_file4 = './raw_data/mesc1/m1_expression3.csv'
# exp_file5 = './raw_data/mesc1/m1_expression4.csv'
# exp_file6 = './raw_data/mesc1/m1_expression5.csv'
# exp_file7 = './raw_data/mesc1/m1_expression6.csv'
# exp_file8 = './raw_data/mesc1/m1_expression7.csv'
# exp_file9 = './raw_data/mesc1/m1_expression8.csv'
tf_file = 'processed_data/'+d_type+'/TF.csv'
target_file = 'processed_data/'+d_type+'/Target.csv'

#data_input = pd.read_csv(exp_file,index_col=0)
# data1_input = pd.read_csv(exp_file1,index_col=0)
# data2_input = pd.read_csv(exp_file2,index_col=0)
# data3_input = pd.read_csv(exp_file3,index_col=0)
# data4_input = pd.read_csv(exp_file4,index_col=0)
# data5_input = pd.read_csv(exp_file1,index_col=0)
# data2_input = pd.read_csv(exp_file2,index_col=0)
# data3_input = pd.read_csv(exp_file3,index_col=0)
# data4_input = pd.read_csv(exp_file4,index_col=0)
#data5_input = pd.read_csv(exp_file5,index_col=0)
#data6_input = pd.read_csv(exp_file6,index_col=0)
#data_input = data1_input.append(data2_input).append(data3_input).append(data4_input).append(data5_input).append(data6_input)
#print(data_input)

# loader1 = load_data(data1_input)
# loader2 = load_data(data2_input)
# loader3 = load_data(data3_input)
# loader4 = load_data(data4_input)
#loader5 = load_data(data5_input)
#loader6 = load_data(data6_input)

# feature1 = loader1.exp_data() # ，1120(gene)*421(cell)
# feature2 = loader2.exp_data()
# feature3 = loader3.exp_data()
# feature4 = loader4.exp_data()
#feature5 = loader5.exp_data()
#feature6 = loader6.exp_data()

#feature = np.array([feature1,feature2,feature3,feature4,feature5,feature6])
#print(feature)
tf = pd.read_csv(tf_file)['index'].values.astype(np.int64) #
target = pd.read_csv(target_file)['index'].values.astype(np.int64) #
# feature1 = torch.from_numpy(feature1)
# feature2 = torch.from_numpy(feature2)
# feature3 = torch.from_numpy(feature3)
# feature4 = torch.from_numpy(feature4)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## 直接用循环
# time_point = 6
# d_type = 'hesc2'
# d_short = 'h2_expression'
data_feature = []
for i in range(1,time_point+1):
    i = i - 1
    exp_file = './processed_data/' + d_type + '/' + d_short + str(int(i)) + '.csv'
    #exp_file = './processed_data/' + d_type + '/' + 'total_mL_expression.csv'
    data_input = pd.read_csv(exp_file, index_col=0)

    # ## 读h5文件
    # store = pd.HDFStore('./processed_data/' + d_type + '/' + d_type_2 + '/' + 'ST_t' + str(i) + '.h5')
    # data_input = store['STrans'].T
    # print(data_input)

    ## log2(pkrm+0.1)
    data_input = data_input.apply(lambda x: np.log2(x + 0.1))

    ##
    loader = load_data(data_input)
    feature = loader.exp_data()
    # print(feature)
    ##
    # pca = PCA(n_components=310)
    # pca_data = pca.fit_transform(feature)

    feature = torch.from_numpy(feature)
    # data_input = np.array(data_input).astype(np.float32)
    # feature = torch.from_numpy(data_input)
    d_feature = feature.to(device)
    #print(d_feature)
    data_feature.append(d_feature)



#feature5 = torch.from_numpy(feature5)
#feature6 = torch.from_numpy(feature6)
tf = torch.from_numpy(tf)

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#feature = [feature1,feature2,feature3,feature4,feature5,feature6]
# data_feature1 = feature1.to(device)
# data_feature2 = feature2.to(device)
# data_feature3 = feature3.to(device)
# data_feature4 = feature4.to(device)
# #data_feature5 = feature5.to(device)
# #data_feature6 = feature6.to(device)
# data_feature = [data_feature1,data_feature2,data_feature3,data_feature4]
tf = tf.to(device)

## data_feature(tensor),tf(tensor),target(numpy)

## 已经划分好的数据集
# train_file = './Train_validation_test/'+data_type+' '+str(num)+'/Train_set.csv'
# test_file = './Train_validation_test/'+data_type+' '+str(num)+'/Test_set.csv'
# val_file = './Train_validation_test/'+data_type+' '+str(num)+'/Validation_set.csv'

train_file = './processed_data/'+d_type+'/Train_set1.csv'
#train_file = './Transfer_learning/'+d_type+'/fine_tune_0.1_r5.csv'
test_file = './processed_data/'+d_type+'/Test_set1.csv'
val_file = './processed_data/'+d_type+'/Validation_set1.csv'

# train_file = 'processed_data/'+"mesc1"+'/Train_set1.csv'
# test_file = 'processed_data/'+"mesc1"+'/Test_set1.csv'
# val_file = 'processed_data/'+"mesc1"+'/Validation_set1.csv'

##
tf_embed_path = r'Result/'+data_type+' '+str(num)+'/Channel1.csv'
target_embed_path = r'Result/'+data_type+' '+str(num)+'/Channel2.csv'
if not os.path.exists('Result/'+data_type+' '+str(num)):
    os.makedirs('Result/'+data_type+' '+str(num))

train_data = pd.read_csv(train_file, index_col=0).values
validation_data = pd.read_csv(val_file, index_col=0).values
test_data = pd.read_csv(test_file, index_col=0).values

##
train_load = scRNADataset(train_data, data_feature[0].shape[0], flag=args.flag)
##
adj = train_load.Adj_Generate(tf,loop=args.loop) # 用train_set组成的矩阵（这是叫做先验知识嘛？？）阿哲
print(adj)

adj = adj2saprse_tensor(adj)
print(adj)
print(adj.size())
## (tf,target,label)
train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
val_data = torch.from_numpy(validation_data)

torch.cuda.empty_cache()

# h1 = 64 h2 = 85 m1 = 150 m2 = 100 mEc2 = 150 mEc3 = 250(best) mEc4 = 80 mEc5 = 150 mEc6 = 175 mEc7 = 128 mEc8 = 100  mEone = 500
# mGMc2 = 150 mGMc3 = 150(best) mGMc4 = 128 mGMc5 = 256 mGMc6 = 150 mGMc7 = 200 mGMc8 = 100  mGMone = 600
# mLc2 = 270(best) mLc3 = 150 mLc4 = 128 mLc5 = 300  mLc6 = 200  mLc7 = 200 mLc8 = 64  mLone =
# mDC = 150 hHEP = 100

model = GTransformer(input_dim= 250, #
                #time_point = args.time_point,
                hidden1_dim=args.hidden_dim[0],
                hidden2_dim=args.hidden_dim[1],
                hidden3_dim=args.hidden_dim[2],
                output_dim=args.output_dim,
                num_head1=args.num_head[0],
                num_head2=args.num_head[1],
                alpha=args.alpha,
                device=device,
                type=args.Type,
                reduction=args.reduction
                #time_point = time_point
                )


##
# device_ids = [0,1,2,3]
# model = torch.nn.DataParallel(model, device_ids=device_ids)

adj = adj.to(device)
model = model.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
validation_data = val_data.to(device)

##
optimizer = Adam(model.parameters(), lr=args.lr)
##
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

model_path = 'model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


##
for epoch in range(args.epochs):
    running_loss = 0.0

    ##
    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()
        # print("取出一批次：{}".format(train_x))
        # print("取出一批次：{}".format(train_y))

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1) # view()相当于reshape()


        # train_y = train_y.to(device).view(-1, 1)
        #model = model.module
        #print("data_feature:{}".format(data_feature))
        pred,tf_feature,target_feature = model(data_feature, adj, train_x) # model.forward(data_feature,adj,train_x)


        #pred = torch.sigmoid(pred)
        if args.flag:
            #pred = torch.softmax(pred, dim=1)
            # print(pred)
            # print(train_y)
            #criterion = torch.nn.CrossEntropyLoss(pred,train_y)
            criterion = torch.nn.CrossEntropyLoss()
            loss_BCE = criterion(pred,train_y)
        else:
            pred = torch.sigmoid(pred)
            loss_BCE = F.binary_cross_entropy(pred, train_y)

        # print("pred:{}".format(pred))
        # print("train_y:{}".format(train_y))

        #loss_BCE = F.binary_cross_entropy(pred, train_y)


        loss_BCE.backward() #
        optimizer.step() #
        scheduler.step() #

        running_loss += loss_BCE.item()

    print('Epoch:{}'.format(epoch + 1),
          'train loss:{}'.format(running_loss))

    model.eval()
    score,tf_feature,target_feature = model(data_feature, adj, validation_data)
    ##
    if args.flag:
        score = torch.softmax(score, dim=1)
        ##
        print(score)
        acc = causal_evaluation(validation_data[:,-1],score)

        print('Epoch:{}'.format(epoch + 1),
              'train loss:{}'.format(running_loss),
              'acc:{:.3F}'.format(acc))

    else:
        score = torch.sigmoid(score)
        # score = torch.sigmoid(score)
        AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=validation_data[:, -1],flag=args.flag)
        #AUC, AUPR, AUPR_norm = Evaluation(y_true=validation_data[:, -1], y_pred=score, flag=args.flag)
            #
        print('Epoch:{}'.format(epoch + 1),
                'train loss:{}'.format(running_loss),
                'AUC:{:.3F}'.format(AUC),
                'AUPR:{:.3F}'.format(AUPR))
##
torch.save(model.state_dict(), model_path + data_type+' '+str(num)+'.pkl')

##
torch.save(model,model_path +model_version + d_type + '_' + str(time_point) + '.pkl')

#
#torch.save(model,'./interpretability/model_new_mesc1.pkl')

model.load_state_dict(torch.load(model_path + data_type+' '+str(num)+'.pkl'))
model.eval()
# tf_embed, target_embed = model.get_embedding()
# print("tf_embed:{}".format(tf_embed))
# print(tf_embed.size())
# embed2file(tf_embed,target_embed,target_file,tf_embed_path,target_embed_path)

score,tf_feature,target_feature = model(data_feature, adj, test_data)
if args.flag:
    score = torch.softmax(score, dim=1)

    ##
    acc = causal_evaluation(test_data[:, -1], score)
    print("测试集acc：{}".format(acc))


else:
    score = torch.sigmoid(score)
    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1],flag=args.flag)
    print('AUC:{}'.format(AUC),
         'AUPRC:{}'.format(AUPR))

