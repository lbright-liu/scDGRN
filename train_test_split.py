import pandas as pd
import numpy as np
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from utils import Network_Statistic
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', type=float, default=0.67, help='the ratio of the training set')
parser.add_argument('--num', type=int, default= 500, help='network scale')
parser.add_argument('--p_val', type=float, default=0.5, help='the position of the target with degree equaling to one')
parser.add_argument('--data', type=str, default='mESC', help='data type')
parser.add_argument('--net', type=str, default='Specific', help='network type')
args = parser.parse_args()




def train_val_test_set(label_file,Gene_file,TF_file,train_set_file,val_set_file,test_set_file,density,p_val=args.p_val):
    #
    #np.random.seed(6)
    #np.random.seed(66)
    #np.random.seed(99)
    #np.random.seed(8)
    #np.random.seed(23)
    #np.random.seed(666)
    #np.random.seed(777)
    #np.random.seed(888)
    #np.random.seed(999)
    #np.random.seed(10000)

    np.random.seed(seeds)

    # hesc2
    #np.random.seed(1221)

    gene_set = pd.read_csv(Gene_file)['index'].values #
    tf_set = pd.read_csv(TF_file)['index'].values #

    label = pd.read_csv(label_file)
    ##
    label_list = label.values.tolist()

    tf = label['TF'].values

    tf_list = np.unique(tf)
    pos_dict = {}
    ##
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    ##
    train_pos = {}
    val_pos = {}
    test_pos = {}
    ##
    for k in pos_dict.keys():
        if len(pos_dict[k]) <= 1:
            p = np.random.uniform(0,1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]

        elif len(pos_dict[k]) == 2:

            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]

        else:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:len(pos_dict[k]) * 2 // 3]
            test_pos[k] = pos_dict[k][len(pos_dict[k]) * 2 // 3:]

            val_pos[k] = train_pos[k][:len(train_pos[k])//5]
            train_pos[k] = train_pos[k][len(train_pos[k])//5:]

##
    train_neg = {}
    for k in train_pos.keys():
        train_neg[k] = []
        for i in range(int(len(train_pos[k])*1/1)):
            neg = np.random.choice(gene_set)
            c = 0
            flag = True
            while neg == k or neg in pos_dict[k] or neg in train_neg[k]:
                neg = np.random.choice(gene_set)
                c = c +1
                if c > 1000:
                    flag = False
                    break
            if flag:
                train_neg[k].append(neg)


    train_pos_set = []
    train_neg_set = []
    for k in train_pos.keys():
        for j in train_pos[k]:
            train_pos_set.append([k, j])
    tran_pos_label = [1 for _ in range(len(train_pos_set))]

    for k in train_neg.keys():
        for j in train_neg[k]:
            train_neg_set.append([k, j])
    tran_neg_label = [0 for _ in range(len(train_neg_set))]



    train_set = train_pos_set + train_neg_set
    train_label = tran_pos_label + tran_neg_label

    train_sample = train_set.copy()
    for i, val in enumerate(train_sample):
        val.append(train_label[i])
    train = pd.DataFrame(train_sample, columns=['TF', 'Target', 'Label'])
    train.to_csv(train_set_file)


    val_pos_set = []
    for k in val_pos.keys():
        for j in val_pos[k]:
            val_pos_set.append([k, j])
    val_pos_label = [1 for _ in range(len(val_pos_set))]

    val_neg = {}
    count = 0
    for k in val_pos.keys():
        val_neg[k] = []
        print(len(val_pos[k]))
        for i in range(int(len(val_pos[k])*1/1)):
            neg = np.random.choice(gene_set)
            c = 0
            flag = True
            while neg == k or neg in pos_dict[k] or neg in train_neg[k] or neg in val_neg[k]:
                neg = np.random.choice(gene_set)
                c = c + 1
                if c > 3000:
                    flag = False
                    break

            if flag:
                val_neg[k].append(neg)
        count = count +1

        print("done{}！！".format(count))


    print("Validation set negative sample collection completed！")
    val_neg_set = []
    for k in val_neg.keys():
        for j in val_neg[k]:
            val_neg_set.append([k,j])


    val_neg_label = [0 for _ in range(len(val_neg_set))]
    val_set = val_pos_set + val_neg_set
    val_set_label = val_pos_label + val_neg_label


    val_set_a = np.array(val_set)
    val_sample = pd.DataFrame()
    val_sample['TF'] = val_set_a[:,0]
    val_sample['Target'] = val_set_a[:,1]
    val_sample['Label'] = val_set_label
    val_sample.to_csv(val_set_file)



    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k, j])

    count = 0
    for k in test_pos.keys():
        count += len(test_pos[k])
    test_neg_num = int(count // density-count)
    test_neg = {}
    for k in tf_set:
        test_neg[k] = []


    ## -----------situation 1--------------------
    test_neg_set = []
    #for i in range(test_neg_num):
    print("test positive count:{}".format(count))
    for i in range(int(count*1/1)):
        t1 = np.random.choice(tf_set)
        t2 = np.random.choice(gene_set)
        c = 0
        flag = True
        while t1 == t2 or [t1, t2] in train_set or [t1, t2] in test_pos_set or [t1, t2] in val_set or [t1,t2] in test_neg_set or [t1,t2] in label_list:
            t2 = np.random.choice(gene_set)
            c = c + 1
            if c > 3000:
                flag = False
                break
        if flag:
            test_neg_set.append([t1, t2])


    ## -----------situation 2----------------------
    # for k in test_pos.keys():
    #     test_neg[k] = []
    #     print(len(test_pos[k]))
    #     for i in range(int(len(test_pos[k])*1/1)):
    #         neg = np.random.choice(gene_set)
    #         c = 0
    #         flag = True
    #         while neg == k or neg in pos_dict[k] or neg in train_neg[k] or neg in val_neg[k] or neg in test_neg[k]:
    #             neg = np.random.choice(gene_set)
    #             c = c + 1
    #             if c > 3000:
    #                 flag = False
    #                 break
    #
    #         if flag:
    #             test_neg[k].append(neg)
    #
    # test_neg_set = []
    # for k in test_neg.keys():
    #     for j in test_neg[k]:
    #         test_neg_set.append([k,j])



        #print("go!")
        # for t1 in tf_set:
        #     for t2 in gene_set:
        #         while t1 != t2 and [t1, t2] not in train_set and [t1, t2] not in test_pos_set and [t1, t2] not in val_set and [t1,t2] not in test_neg_set:
        #             test_neg_set.append([t1,t2])


    test_pos_label = [1 for _ in range(len(test_pos_set))]
    test_neg_label = [0 for _ in range(len(test_neg_set))]

    test_set = test_pos_set + test_neg_set
    #test_set = test_pos_set
    test_label = test_pos_label + test_neg_label
    for i, val in enumerate(test_set):
        val.append(test_label[i])

    test_sample = pd.DataFrame(test_set, columns=['TF', 'Target', 'Label'])
    test_sample.to_csv(test_set_file)




if __name__ == '__main__':
    # data_type = args.data
    # net_type = args.net
    #data_type = 'hesc1'
    #data_type = 'mouse_brain/pretrain'
    #data_type = 'Zma_NU_data'
    #density = Network_Statistic(data_type=data_type, net_scale=args.num, net_type=net_type)
    density = 19189/(36*19189)

    r_list = [6,7,8,9,10]
    seed_list = [666,777,888,999,10000]
    for data_type in ['hesc2','mesc1','mesc2','mHSC-E','mHSC-GM','mHSC-L']:
        for rounds in range(5):
            TF2file = './processed_data/'+data_type+'/TF.csv'
            Gene2file = './processed_data/'+data_type+'/Target.csv'
            #Gene2file = os.getcwd() + '/' + net_type + ' Dataset/' + data_type + '/TFs+' + str(args.num) + '/Target.csv'
            label_file = 'processed_data/'+data_type+'/label.csv'


            train_set_file = 'processed_data/'+data_type+'/Train_set' + str(r_list[rounds]) + '.csv'
            test_set_file = 'processed_data/'+data_type+'/Test_set' + str(r_list[rounds]) + '.csv'
            val_set_file = 'processed_data/'+data_type+'/Validation_set' + str(r_list[rounds]) + '.csv'
            seeds = seed_list[rounds]

            # path = os.getcwd() + '/.../' + net_type + '/' + data_type + ' ' + str(args.num)
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # if net_type == 'Specific':
            #     Hard_Negative_Specific_train_test_val(label_file, Gene2file, TF2file, train_set_file, val_set_file,
            #                                           test_set_file)
            #else:
            train_val_test_set(label_file, Gene2file, TF2file, train_set_file, val_set_file, test_set_file, density)


