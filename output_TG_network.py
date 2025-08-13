import math

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
#from gRNN import GRNN
from torch.optim.lr_scheduler import StepLR
#import scipy.sparse as sp
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,causal_evaluation
import pandas as pd
#from torch.utils.tensorboard import SummaryWriter
#from Code.PytorchTools import EarlyStopping
import numpy as np
import random
#import glob
import os
from sklearn import metrics
import matplotlib.pyplot as plt
#import time
import argparse
import networkx as nx
import community

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# time_point = 4
# d_type = 'Zma_NU_data'
# d_short = 'Zma_expression'
# model_version = 'Zma_causal_'
#
time_point = 6
d_type = 'hesc2'
#d_type = 'mHSC-E'
d_short = 'h2_expression'
#d_short = 'mEc3_expression'
model_version = 'v7_causal_'

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
#parser.add_argument('--epochs', type=int, default= 90, help='Number of epoch.')
parser.add_argument('--epochs', type=int, default= 5, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3,3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=512, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type',type=str,default='dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')

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



net_type = "Specific"


density = 1/36

tf_file = 'processed_data/'+d_type+'/TF.csv'
target_file = 'processed_data/'+d_type+'/Target.csv'


tf = pd.read_csv(tf_file)['index'].values.astype(np.int64) #
target = pd.read_csv(target_file)['index'].values.astype(np.int64) #

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_feature = []
for i in range(time_point):
    exp_file = './processed_data/' + d_type + '/' + d_short + str(int(i)) + '.csv'
    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()
    feature = torch.from_numpy(feature)
    d_feature = feature.to(device)
    data_feature.append(d_feature)

    # exp_file = './processed_data/' + d_type + '/' + d_short + str(int(i)) + '.csv'
    # # exp_file = './processed_data/' + d_type + '/' + 'total_h2_expression.csv'
    # data_input = pd.read_csv(exp_file, index_col=0)
    #
    # ## 随机dropout  4%  8%
    # print(data_input.shape[1])
    #
    # dropout_rate = 0.4  # 设置丢弃率为20% 40%
    #
    # num_cells = data_input.shape[1]  # 细胞的数量
    # num_cells_dropout = int(num_cells * dropout_rate)  # 需要丢弃的细胞数量
    #
    # # 随机选择要丢弃的细胞的索引
    # dropout_indices = np.random.choice(num_cells, size=num_cells_dropout, replace=False)
    #
    # # 通过删除对应的列来丢弃细胞
    # data_input = data_input.drop(columns=data_input.columns[dropout_indices])
    # print(data_input.shape[1])
    #
    # # ## 读h5文件
    # # store = pd.HDFStore('./processed_data/' + d_type + '/' + d_type_2 + '/' + 'ST_t' + str(i) + '.h5')
    # # data_input = store['STrans'].T
    # # print(data_input)
    #
    # ## log2(pkrm+0.1)
    # data_input = data_input.apply(lambda x: np.log2(x + 0.1))
    #
    # ## 对数据进行了标准化
    # loader = load_data(data_input)
    # feature = loader.exp_data()
    # # print(feature)
    # ## pca降维
    # # pca = PCA(n_components=310)
    # # pca_data = pca.fit_transform(feature)
    #
    # feature = torch.from_numpy(feature)
    # # data_input = np.array(data_input).astype(np.float32)
    # # feature = torch.from_numpy(data_input)
    # d_feature = feature.to(device)
    # # print(d_feature)
    # data_feature.append(d_feature)



#feature5 = torch.from_numpy(feature5)
#feature6 = torch.from_numpy(feature6)
tf = torch.from_numpy(tf)
tf = tf.to(device)


test_file = 'processed_data/'+d_type+'/Test_set1.csv'
train_file = 'processed_data/'+d_type+'/Train_set1.csv'
val_file = 'processed_data/'+d_type+'/Validation_set1.csv'

##
tf_embed_path = r'Result/'+data_type+' '+str(num)+'/Channel1.csv'
target_embed_path = r'Result/'+data_type+' '+str(num)+'/Channel2.csv'
if not os.path.exists('Result/'+data_type+' '+str(num)):
    os.makedirs('Result/'+data_type+' '+str(num))

test_data = pd.read_csv(test_file, index_col=0).values
train_data = pd.read_csv(train_file, index_col=0).values
val_data = pd.read_csv(val_file, index_col=0).values
##
train_load = scRNADataset(train_data, data_feature[0].shape[0], flag=args.flag)
##
adj = train_load.Adj_Generate(tf,loop=args.loop) #

adj = adj2saprse_tensor(adj)

## (tf,target,label)
test_data = torch.from_numpy(test_data)
val_data = torch.from_numpy(val_data)
torch.cuda.empty_cache()

adj = adj.to(device)
#model = model.to(device)
test_data = test_data.to(device)
val_data = val_data.to(device)

model_path = 'model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

########################### Predict the output using the loaded model
#

#gene = [1410,14234,9896,9623,14623,6133,14229,8377,5891,11783]  ## has pathyway gene


##
df = pd.read_csv('./processed_data/hesc2/gene_index1000.csv')
#df = pd.read_csv('./processed_data/mHSC-E/Target.csv')

gene = df['gene_index'].tolist()
#gene = df['index'].tolist()
print("A list of genes that have undergone quality control：{}".format(gene))


# 生成exp_set.csv
df_tf_all = pd.read_csv("./processed_data/hesc2/TF.csv")
#df_tf_all = pd.read_csv("./processed_data/mHSC-E/TF.csv")

#df_target_all = pd.read_csv("./Transfer_learning/mouse_brain/Target.csv")

tf_all = df_tf_all['index'].tolist()
target_all = gene

##
# tf_all = [9896,14623,11783] # [nanog,srf,pou5f1]  srf,pou5f1,bhlhe40,
# target_all = [1410,14234,9623,6133,14229,8377,5891] # [bhlhe40,smad7,msx1,gata3,smad2,lef1,foxh1]




#gene = gene[:1000]

#gene = [1410,14234,9896,9623,14623,6133,14229,8377,5891,11783]

# generate exp_set.csv
exp_data = []
for i in range(len(tf_all)):
    for j in range(len(target_all)):
        data = [tf_all[i],target_all[j]]
        exp_data.append(data)
exp_set = pd.DataFrame(exp_data,columns=['TF', 'Target'])
exp_set.to_csv('processed_data/'+d_type+'/exp_set_TG.csv')
exp_data = pd.read_csv('processed_data/'+d_type+'/exp_set_TG.csv', index_col=0).values
exp_data = torch.from_numpy(exp_data)
exp_data = exp_data.to(device)

print("exp_data:{}".format(exp_data))
# exp_data1 = exp_data[:10000]
# exp_data2 = exp_data[10000:]
exp_data1 = exp_data[:5]
exp_data2 = exp_data[5:]
print("exp_data1:{}".format(exp_data1))

model = torch.load('./model/'+'TG_hesc2_demo.pkl')
model = model.to(device)
model.eval()

for recons_tp in range(1,7):

    score,tf_feature,target_feature = model(data_feature, adj, exp_data,recons_tp)
    df_exp = None
    if args.flag:
        score = torch.softmax(score, dim=1)

        ##
        acc = causal_evaluation(test_data[:, -1], score)
        print("test set acc：{}".format(acc))


    else:
        score = torch.sigmoid(score)
        #print(score)


        score = score.flatten().tolist()
        #print(score)

        ## 保存预测结果
        df_exp = pd.read_csv('processed_data/' + d_type + '/exp_set_TG.csv', index_col=0)
        df_exp['score'] = score
        #print(df_exp)
        #df_exp.to_csv('processed_data/' + d_type + '/TG_data/hesc2_regulate_LessTrain_0.8_prediction1.csv',index=False)
        #df_exp.to_csv('processed_data/' + d_type + '/TG_data/hesc2_regulate_prediction6.csv', index=False)
        #df_exp.to_csv('processed_data/' + d_type + '/TG_data/mHSC-E_regulate_prediction1.csv', index=False)
        #df_exp.to_csv('processed_data/' + d_type + '/TG_data/hesc2_regulate_LessCell_0.6_prediction1.csv', index=False)



    ##----------- read csv, obtain top 20% high-score regulatory edge----------------------------------

    import pandas as pd

    #
    #df = pd.read_csv('./hesc2_regulate_mislabel_all_to_one.csv')  #
    #df = pd.read_csv('./hesc2_regulate_LessTrain_0.8_prediction1.csv')
    #df = pd.read_csv('./hesc2_regulate_mislabel_prediction1.csv')
    #df = pd.read_csv('./hesc2_regulate_mislabel_GENELink.csv')
    #df = pd.read_csv('./hesc2_regulate_LessTrain0.8_GENELink.csv')
    #df = pd.read_csv('./hesc2_regulate_LessCell_0.6_prediction6.csv')

    df = df_exp
    # Sort in descending order according to the 'score' column
    df_sorted = df.sort_values(by='score', ascending=False)

    #
    top_percent = 0.2  # Select the top 20% of rows
    num_rows = int(len(df_sorted) * top_percent)

    # Select the top 20% rows with the highest scores
    df_selected = df_sorted.head(num_rows)
    print("t{} is {}:".format(recons_tp,df_selected))
    #
    #df_selected.to_csv('./hesc2_regulatory_t6_sp.csv', index=False)  #
    #df_selected.to_csv('./hesc2_regulatory_mislabel_all_to_one.csv', index=False)
    df_selected.to_csv('processed_data/' + d_type + '/hesc2_regulatory_t' + str(recons_tp) + '.csv', index=False)
    #df_selected.to_csv('./hesc2_regulatory_mislabel_t1.csv', index=False)
    #df_selected.to_csv('./hesc2_regulatory_mislabel_GENELink.csv', index=False)
    #df_selected.to_csv('./hesc2_regulatory_LessTrain0.8_GENELink.csv', index=False)
    #df_selected.to_csv('./hesc2_regulatory_LessCell_0.6_t6.csv', index=False)

print("The restructured dynamic gene regulatory network has been saved in : /processed_data/hesc2")