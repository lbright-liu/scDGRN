import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
from torch.nn import CosineSimilarity
import math

class GRNN(nn.Module):
    def __init__(self,input_dim,hidden1_dim,hidden2_dim,hidden3_dim,output_dim,num_head1,num_head2,
                 alpha,device,type,reduction):
        super(GRNN, self).__init__()
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.device = device
        self.alpha = alpha
        self.type = type
        self.reduction = reduction
        self.time_point = 3
        self.flag = False

        # self.attention = nn.Sequential(
        #     nn.Linear(128, 1),
        #     nn.Tanh(),
        #     nn.Softmax(dim=1)
        # )
        self.batch_size = 512
        self.timesteps = 3
        self.embed_dim = 16
        self.num_heads = 2

        #self.units = 128 #准确率0.69
        self.units = 64 # 64比128效果更好，测试准确率为0.710


        #self.s_attention = SelfAttention(self.embed_dim, self.num_heads)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.self_attention_layer1 = SelfAttention(self.embed_dim, self.num_heads).to(device)
        self.self_attention_layer2 = SelfAttention(self.embed_dim, self.num_heads).to(device)
        self.self_attention_layer3 = SelfAttention(self.embed_dim, self.num_heads).to(device)
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.weighted_fc = nn.Linear(self.embed_dim * 2, 1)  # 定义自适应权重层

        #self.w_rs = nn.Parameter(torch.tensor(0.1))
        self.init_value = 0.01
        #self.w_rs = nn.Parameter(torch.ones(1, 1, 32) * self.init_value)
        #self.w_rs = nn.Parameter(torch.tensor([self.init_value]))
        self.w = nn.Parameter(0.01*torch.ones(2))

        #self.position = PositionLearnable(32,4)
        #self.multihead_attn = nn.MultiheadAttention(self.embed_dim,self.num_heads)

        #self.tansput = nn.Linear(self.units * self.timesteps, 1)
        self.tansput = nn.Linear(32 * self.timesteps, 1)  ## 训练的时候存储的就是四个时间点的参数，为128
        #self.tansput = nn.Linear(64, 1)
        self.fc1 = nn.Linear(self.embed_dim, self.units)
        self.fc2 = nn.Linear(self.units, self.units)


        if self.reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
        elif self.reduction == 'concate':
            self.hidden1_dim = num_head1*hidden1_dim
            self.hidden2_dim = num_head2*hidden2_dim

        ## 怎么每一层gat都是三个头部，并且没看到平均啊
        ## gat1
        self.ConvLayer1 = [AttentionLayer(input_dim,hidden1_dim,alpha) for _ in range(num_head1)]
        for i, attention in enumerate(self.ConvLayer1):
            ## ConvLayer1_AttentionHead1，ConvLayer1_AttentionHead2，ConvLayer1_AttentionHead3
            self.add_module('ConvLayer1_AttentionHead{}'.format(i),attention)
        ## gat2
        self.ConvLayer2 = [AttentionLayer(self.hidden1_dim,hidden2_dim,alpha) for _ in range(num_head2)]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module('ConvLayer2_AttentionHead{}'.format(i),attention)

        ## tf和target分开放入一个mlp里面
        ## MLP1
        self.tf_linear1 = nn.Linear(hidden2_dim,hidden3_dim)
        self.target_linear1 = nn.Linear(hidden2_dim,hidden3_dim)
        ## MLP2
        self.tf_linear2 = nn.Linear(hidden3_dim,output_dim)
        self.target_linear2 = nn.Linear(hidden3_dim, output_dim)

        ## 输入维度统一  特征重构
        # hesc2
        # self.nomal_linear1 = nn.Linear(92, input_dim)
        # self.nomal_linear2 = nn.Linear(102, input_dim)
        # self.nomal_linear3 = nn.Linear(66, input_dim)
        # self.nomal_linear4 = nn.Linear(172, input_dim)
        # self.nomal_linear5 = nn.Linear(138, input_dim)
        # self.nomal_linear6 = nn.Linear(188, input_dim)
        # mesc1
        #self.nomal_linear = nn.Linear(384, input_dim)
        # hesc1

        # self.nomal_linear1 = nn.Linear(81, input_dim)
        # self.nomal_linear2 = nn.Linear(190, input_dim)
        # self.nomal_linear3 = nn.Linear(377, input_dim)
        # self.nomal_linear4 = nn.Linear(415, input_dim)
        # self.nomal_linear5 = nn.Linear(466, input_dim)
        # mesc2
        # self.nomal_linear1 = nn.Linear(933, input_dim)
        # self.nomal_linear2 = nn.Linear(303, input_dim)
        # self.nomal_linear3 = nn.Linear(683, input_dim)
        # self.nomal_linear4 = nn.Linear(798, input_dim)

        # s1
        # self.nomal_linear1 = nn.Linear(2078, input_dim)
        # self.nomal_linear2 = nn.Linear(1991, input_dim)
        # self.nomal_linear3 = nn.Linear(1992, input_dim)
        # self.nomal_linear4 = nn.Linear(1991, input_dim)
        # self.nomal_linear5 = nn.Linear(1991, input_dim)
        # self.nomal_linear6 = nn.Linear(1992, input_dim)
        # self.nomal_linear7 = nn.Linear(1991, input_dim)
        # self.nomal_linear8 = nn.Linear(1991, input_dim)
        # self.nomal_linear9 = nn.Linear(1992, input_dim)
        # self.nomal_linear10 = nn.Linear(1991, input_dim)

        # s2
        # self.nomal_linear1 = nn.Linear(4150, input_dim)
        # self.nomal_linear2 = nn.Linear(3218, input_dim)
        # self.nomal_linear3 = nn.Linear(2631, input_dim)
        # self.nomal_linear4 = nn.Linear(2318, input_dim)
        # self.nomal_linear5 = nn.Linear(2001, input_dim)
        # self.nomal_linear6 = nn.Linear(2027, input_dim)
        # self.nomal_linear7 = nn.Linear(1945, input_dim)
        # self.nomal_linear8 = nn.Linear(1710, input_dim)

        # Z_ma
        # self.nomal_linear1 = nn.Linear(5, input_dim)
        # self.nomal_linear2 = nn.Linear(7, input_dim)
        # self.nomal_linear3 = nn.Linear(11, input_dim)
        # self.nomal_linear4 = nn.Linear(8, input_dim)


        # mEc3（best）
        # self.nomal_linear1 = nn.Linear(495, input_dim)
        # self.nomal_linear2 = nn.Linear(335, input_dim)
        # self.nomal_linear3 = nn.Linear(241, input_dim)

        # # mGMc3（best）
        self.nomal_linear1 = nn.Linear(279, input_dim)
        self.nomal_linear2 = nn.Linear(321, input_dim)
        self.nomal_linear3 = nn.Linear(289, input_dim)

        # # mLc2 （best）
        # self.nomal_linear1 = nn.Linear(425, input_dim)
        # self.nomal_linear2 = nn.Linear(422, input_dim)


        ## 输出层
        if self.type == 'MLP':
            self.linear = nn.Linear(2*output_dim, 2) ## 两个低维表示做连接后输入到这层然后输出两个分类节点

        ## lstm层
        # self.rnn = nn.LSTM(  # LSTM 效果要比 nn.RNN() 好多了
        #     input_size=32,  # 图片每行的数据像素点
        #     hidden_size=256,  # rnn hidden unit
        #     num_layers=2,  # 有几层 RNN layers
        #     batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        # )

        ## gru层
        self.gru = nn.GRU(input_size=32, hidden_size=128, num_layers=2, batch_first=True)


        self.attention = Attention(self.units) #最后的多时间点加权融合注意力层 32表示输入的特征向量维度

        #self.tf_ecoder = TransformerEcoder()
        #self.mutil_attention = MultiHeadAttentionLayer(inputs)

        #self.out = nn.Linear(128, 1)  # 输出层
        # self.out = nn.Linear(128,10)
        # self.fout = nn.Linear(10,1)

        #self.out = nn.Linear(128, 16)
        self.final_out = nn.Linear(self.units,16)
        self.final_out_causal = nn.Linear(2*self.units,3)
        self.final_out_causal_two_label = nn.Linear(2 * self.units, 1)


        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters() ## attentionlayer类里面的函数

        for attention in self.ConvLayer2:
            attention.reset_parameters()

        nn.init.xavier_uniform_(self.tf_linear1.weight,gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)




    def encode(self,x,adj):
        ## 看att(x,adj)的输出维度吧，如果是一列，下面的操作就很迷？？盲猜是一行
        if self.reduction =='concate':
            x = torch.cat([att(x, adj) for att in self.ConvLayer1], dim=1) # 对列进行拼接，dim=0是对行进行拼接
            x = F.elu(x)

        elif self.reduction =='mean':
            x = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer1]), dim=0) # 相当于竖着拼接成的一条向量所有元素求均值？
            x = F.elu(x)

        else:
            raise TypeError


        with torch.no_grad():
            out = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer2]),dim=0) #修改过，加了一个梯度消除，不然内存不够

        return out


    def decode(self,tf_embed,target_embed):

        # if self.type =='dot':
        #     ## mul()两个张量对应元素相乘
        #     prob = torch.mul(tf_embed, target_embed)
        #     prob = torch.sum(prob,dim=1).view(-1,1) ##整个batch的预测结果，是一列tensor
        #
        #
        #     return prob
        #
        # elif self.type =='cosine':
        #     prob = torch.cosine_similarity(tf_embed,target_embed,dim=1).view(-1,1)
        #
        #     return prob
        #
        # elif self.type == 'MLP':
        #     h = torch.cat([tf_embed, target_embed],dim=1)
        #     prob = self.linear(h)
        #
        #     return prob
        # else:
        #     raise TypeError(r'{} is not available'.format(self.type))
        t_p = self.time_point
        #h_t = []
        h_t_tf = []
        h_t_target = []
        for i in range(t_p):
            # h1 = torch.cat([tf_embed[i], target_embed[i]], dim=1)
            # h_t.append(h1)

            h1 = tf_embed[i]
            h2 = target_embed[i]
            h_t_tf.append(h1)
            h_t_target.append(h2)
        # h1 = torch.cat([tf_embed[0], target_embed[0]], dim=1)
        # h2 = torch.cat([tf_embed[1], target_embed[1]], dim=1)
        # h3 = torch.cat([tf_embed[2], target_embed[2]], dim=1)
        # h4 = torch.cat([tf_embed[3], target_embed[3]], dim=1)
        #h5 = torch.cat([tf_embed[4], target_embed[4]], dim=1)
        #h6 = torch.cat([tf_embed[5], target_embed[5]], dim=1)
        #h_x = torch.cat(h_t, dim = 1).view(-1,t_p,16*2)
        h_x_tf = torch.cat(h_t_tf, dim=1).view(-1, t_p, 16)
        h_x_target = torch.cat(h_t_target, dim=1).view(-1, t_p, 16)
        #prob = self.linear(h)
        ## 放到lstm里面去
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        #r_out, (h_n, h_c) = self.gru(h_x, None)  # None 表示 hidden state 会用全0的 state 就是128维


        #inputs = torch.randn(batch_size, timesteps, embed_dim)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        inputs_tf = h_x_tf  # (512,time_point,32)
        inputs_target = h_x_target
        # 定义位置编码1
        # pos_encoding = np.zeros((1, t_p, self.embed_dim)) # 这里是1,和input相加时会在0维上扩张然后相加，相当于复制，每个样本数据都会加上这个位置编码值
        # #pos_encoding = torch.tensor(pos_encoding)
        # #pos_encoding = pos_encoding.to(device)
        # for i in range(t_p):
        #     for j in range(self.embed_dim):
        #         pos_encoding[0, i, j] = np.sin(i / (10000 ** ((2 * j) / self.embed_dim)))
        # pos_encoding = torch.tensor(pos_encoding, dtype=torch.float32)
        # pos_encoding = pos_encoding.to(device)

        ## 位置编码2
        pos_encoding = torch.zeros((t_p, self.embed_dim))
        pos = torch.arange(0, t_p, dtype=torch.float).unsqueeze(1)
        #div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim))  ## 可以通过调整位置编码中的除数来控制粒度。具体地说，除数的大小越小，粒度就越细。时间点很少的话应该要调小除数
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim))
        pos_encoding[:, 0::2] = torch.sin(pos * div_term)
        pos_encoding[:, 1::2] = torch.cos(pos * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        pos_encoding = pos_encoding.to(device)


        x_pos_tf = inputs_tf + pos_encoding  ## 这样相加合理吗
        x_pos_target = inputs_target + pos_encoding
       # x = inputs
        #x = self.position(inputs)  ## 这样连接得到得x不是32维了，肯定大于32维，应该是64维
        x_tf = x_pos_tf.to(device)
        x_target = x_pos_target.to(device)
        # 应用多头注意力机制层
        #multihead = nn.MultiheadAttention(self.embed_dim,self.num_heads).to(device)
        #attn_output, attn_weights = self.multihead_attn(x, x, x)
        #attn_output, attn_weights = multihead(x,x,x)

        ## 单层自注意力机制，多头平均
        # attention = MultiHeadAttentionLayer(self.embed_dim, self.num_heads).to(device)
        # x = attention(x)

        ##多层自注意力，多头连接
        #input = inputs
        x_tf = self.self_attention_layer1(x_tf)
        x_target = self.self_attention_layer1(x_target)
        # x_tf = self.self_attention_layer2(x_tf)
        # x_target = self.self_attention_layer2(x_target)
        # x_tf = self.self_attention_layer3(x_tf)
        # x_target = self.self_attention_layer3(x_target)
        #x = self.self_attention_layer2(x)
        #x = self.self_attention_layer3(x)
        #output = 16 * input + x
        ## 相当于是有一个权重（未归一化）
        #output = 8*x_pos + 1*x

        ## 自适应权重
        # 按元素拼接 a 和 b
        # c = torch.cat((x_pos, x), dim=2)
        # # 计算每个元素的权重
        # w = torch.sigmoid(self.weighted_fc(c))
        # # 加权求和
        # output = w * x_pos + (1 - w) * x

        ## 进阶版自适应权重
        #w = self.w_rs.expand(-1, 32)
        #w = self.w_rs.expand(x_pos.size())
        #w = self.w_rs
        w_rs = 100*self.w
        w1 = torch.exp(w_rs[0]) / torch.sum(torch.exp(w_rs))
        w2 = torch.exp(w_rs[1]) / torch.sum(torch.exp(w_rs))

        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))

        #print("更新w：{}".format(w_rs))
        #w_ = w*800
        #print("此时的w_:{}".format(w_))
        # 加权求和
        #output = w1 * x_pos + w2 * x
        #output = 0 * x_pos + 1 * x
        output_tf = 1 * inputs_tf + 1 * x_tf
        output_target = 1 * inputs_target + 1 * x_target
        # output = self.layer_norm(inputs + output)
        x_tf = self.layer_norm(output_tf)
        x_target = self.layer_norm(output_target)

        # prob = torch.mul(x_tf[:,-1,:], x_target[:,-1,:])
        # prob = torch.sum(prob,dim=1).view(-1,1) ##整个batch的预测结果，是一列tensor
        # return prob

        ## transformerecoder
        #x = self.tf_ecoder(x)

        # 全连接层
        x_tf = F.relu(self.fc1(x_tf))         ## 把32维的特征向量转化到128维的隐藏空间，再进行多个时间点的融合注意力
        x_target = F.relu(self.fc1(x_target))


        #x = F.relu(self.fc2(x)) ## (512,4,64) 无意义的转换
        #print(x.size())
        # 将六个时间点的输出特征向量拼接在一起
        #x = x.view(x.size()[0], -1) ## (512,64*4)
        # print(x)
        # print(x.size())
        # print(x)
        # print(x.size())
        # 定义二分类任务的全连接层
        #output_layer = nn.Linear(self.units * self.timesteps, 1)
        #output = self.tansput(x)
        # print(output)
        # print(output.size())



        ## 最有一层注意力融合多个时间点
        #attention_output = self.attention(r_out)

        attention_output_tf = self.attention(x_tf)  ## 如果不进行融合的话就直接取x的最后一个时间点的向量，然后于另外一个x最后一个向量做点积得到score
        attention_output_target = self.attention(x_target)



        #print(attention_output_tf.size())

        ## 因果推断（准备再并入一个差异特征向量，和tf,target拼接）
        if self.flag:
            liner_concat = torch.cat((attention_output_tf,attention_output_target),dim=1)
            #print(liner_concat.size())
            prob = self.final_out_causal(liner_concat)
            return prob

        else:
            # linear_output_tf = self.final_out(attention_output_tf)
            # #linear_output_tf = F.leaky_relu(linear_output_tf)
            # linear_output_target = self.final_out(attention_output_target)
            # #linear_output_target = F.leaky_relu(linear_output_target)
            #
            # ##保存最后的特征向量
            # self.tf_ouput = linear_output_tf
            # self.target_output = linear_output_target
            #
            # prob = torch.mul(linear_output_tf, linear_output_target)
            # prob = torch.sum(prob,dim=1).view(-1,1) ##整个batch的预测结果，是一列tensor



            ########### two_label_causal
            liner_concat = torch.cat((attention_output_tf, attention_output_target), dim=1)
            # print(liner_concat.size())
            prob = self.final_out_causal_two_label(liner_concat)


            return prob,attention_output_tf,attention_output_target

        # 这个地方选择lstm_output[-1]，也就是相当于最后一个输出，因为其实每一个cell（相当于图中的A）都会有输出，但是我们只关心最后一个
        # 选取最后一个时间点的 r_out 输出 r_out[:, -1, :] 的值，也是 h_n 的值
        # torch.Size([64, 28, 128])->torch.Size([64,128])
        #out = self.out(r_out[:,-1, :])  # torch.Size([64, 128])-> torch.Size([64, 10])    ##这里应该是256*2
        #out = F.relu(out)
        #out = F.dropout(out,p=0.5)
        #out = self.final_out(out)
        #out = F.leaky_relu(out)

        ## 多输出
        # l_out = []
        # for i in range(t_p):
        #     out = self.out(r_out[:, i, :])
        #     out = F.relu(out)
        #     l_out.append(out)
        # ll_out = torch.cat(l_out, dim=1)
        # #ll_out = self.out(ll_out)
        # #F.dropout(ll_out,p=0.01)
        # lll_out = self.final_out(ll_out)

        ## 池化操作
        # l_out = []
        # for i in range(t_p):
        #     out = self.out(r_out[:, i, :])
        #     #out = F.relu(out)
        #     #out = out.cpu().detach().numpy()
        #     l_out.append(out)
        #
        # feature_vectors = l_out
        # # 将所有特征向量按列连接成一个二维矩阵
        # features_matrix = torch.stack(feature_vectors,dim=1)
        # pooled_features = torch.mean(features_matrix,dim=1)
        #
        # lll_out = self.final_out(pooled_features)
        #return prob

    ## 将训练数据输入网络求解pred
    def forward(self,x,adj,train_sample):
        ## 经过两层gat后的特征向量
        train_tf_total = []
        train_target_total = []
        for i in range(self.time_point):
            if i == 0:
                e = self.nomal_linear1(x[i])
            elif i == 1:
                e = self.nomal_linear2(x[i])
            # elif i == 2:
            #     e = self.nomal_linear3(x[i])
            # elif i == 3:
            #     e = self.nomal_linear4(x[i])
            # elif i == 4:
            #     e = self.nomal_linear5(x[i])
            # elif i == 5:
            #     e = self.nomal_linear6(x[i])
            # elif i == 6:
            #     e = self.nomal_linear7(x[i])
            # elif i == 7:
            #     e = self.nomal_linear8(x[i])
            # elif i == 8:
            #     e = self.nomal_linear9(x[i])
            else:
                e = self.nomal_linear3(x[i])
            #e = self.nomal_linear(x[i])
            embed = self.encode(e,adj)
            #print(embed)
            #embed = self.encode(x[i], adj)

            tf_embed = self.tf_linear1(embed)
            tf_embed = F.leaky_relu(tf_embed)
            tf_embed = F.dropout(tf_embed,p=0.01)
            tf_embed = self.tf_linear2(tf_embed)
            tf_embed = F.leaky_relu(tf_embed)

            target_embed = self.target_linear1(embed)
            target_embed = F.leaky_relu(target_embed)
            target_embed = F.dropout(target_embed, p=0.01)
            target_embed = self.target_linear2(target_embed)
            target_embed = F.leaky_relu(target_embed)

            ## 这个是所有的基因经过网络层的低维向量化表示
            # self.tf_ouput = tf_embed
            # self.target_output = target_embed

            ## 提取出这个batch的数据，tf,target分离
            train_tf = tf_embed[train_sample[:,0]] # 取得是tf基因对应的特征向量（这个batch的所有的tf）
            train_target = target_embed[train_sample[:, 1]] # 取得是target基因对应的特征向量
            train_tf_total.append(train_tf)
            train_target_total.append(train_target)
        ## 返回最终lstm输出的结果
        pred = self.decode(train_tf_total, train_target_total)

        return pred

    def get_embedding(self):
        return self.tf_ouput, self.target_output



class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 初始化查询、键、值权重矩阵
        self.query_weight = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.key_weight = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.value_weight = nn.Linear(d_model, d_model * n_heads, bias=False)

        # 初始化输出权重矩阵
        self.output_weight = nn.Linear(d_model * n_heads, d_model)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]

        batch_size, seq_len, d_model = x.size()

        # 通过查询、键、值权重矩阵生成查询、键、值张量
        query = self.query_weight(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        key = self.key_weight(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        value = self.value_weight(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_v]

        # 计算注意力分数
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))  # [batch_size, n_heads, seq_len, seq_len]

        # 屏蔽掉无效位置
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 使用注意力权重加权求和
        attn_output = torch.matmul(attn_weights, value)  # [batch_size, n_heads, seq_len, d_v]

        # 将注意力头拼接起来
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # [batch_size, seq_len, n_heads * d_v]

        # 使用输出权重矩阵变换
        output = self.output_weight(attn_output)  # [batch_size, seq_len, d_model]

        return output


# class ResidualBlock(nn.Module):
#     def __init__(self, d_model, n_heads, dropout_rate=0.1):
#         super(ResidualBlock, self).__init__()
#         self.d_model = d_model
#
#         # 定义第一层自注意力层
#         self.self_attention_layer1 = SelfAttention(d_model, n_heads)
#
#         # 定义第二层自注意力层
#         self.self_attention_layer2 = SelfAttention(d_model, n_heads)
#         self.layer_norm = nn.LayerNorm(d_model)
#
#         # 定义两层自注意
#     def forward(self,x):
#         x1 = self.self_attention_layer1(x)
#         x2 = self.self_attention_layer2(x1)
#         output = x2
#         output = x + output
#         output = self.layer_norm(output)
#         return output

# 定义多头自注意力机制层
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query_fc = nn.Linear(embed_dim, embed_dim)
        self.key_fc = nn.Linear(embed_dim, embed_dim)
        self.value_fc = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs):  ## (512,time_steps,32)
        batch_size = inputs.size(0)
        query = self.query_fc(inputs).view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2) ## 置换第二个和第三个维度
        key = self.key_fc(inputs).view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
        value = self.value_fc(inputs).view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim // self.num_heads)
        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)

        context_vector = torch.matmul(attention_weights, value)
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # output = self.fc(context_vector)
        # output = self.dropout(output)
        output = context_vector
        output = 4*inputs + output
        #output = self.layer_norm(inputs + output)
        output = self.layer_norm(output)
        return output


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim # gru隐藏层输出128维
        self.linear = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # inputs: batch_size x seq_length x input_dim
        # 对输入进行线性变换
        linear_output = self.linear(inputs)
        # 计算注意力分布
        attention_weights = self.softmax(linear_output)
        # 利用注意力分布加权求和
        attention_output = torch.sum(attention_weights * inputs, dim=1)
        return attention_output


class AttentionLayer(nn.Module):
    def __init__(self,input_dim,output_dim,alpha=0.2,bias=True):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim #细胞个数（一个基因的维度），421
        self.output_dim = output_dim #16
        self.alpha = alpha

        ## w = (output_dim*input_dim)
        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim,self.output_dim))
        ## a的维度是多少？？应该是1*2F
        self.a = nn.Parameter(torch.zeros(size=(2*self.output_dim,1))) # g1,g2做连接后与a相乘


        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        ## 初始化参数，保持每层输入和输出的方差相同
        self.reset_parameters()


    def reset_parameters(self):
        ## 初始化，xavier:通过网络层时，保持输入和输出的方差相同
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, x):

        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        e = F.leaky_relu(Wh1 + Wh2.T,negative_slope=self.alpha)
        return e

    ############（data_feature,adj）att()
    def forward(self,x,adj):
        ## matmul()若两个tensor都是一维的，则返回两个向量的点积运算结果；若不是则返回两个矩阵相乘结果
        h = torch.matmul(x, self.weight) # x:1120*421   weight:16*421
        e = self._prepare_attentional_mechanism_input(h) ## 在

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense()>0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.softmax(e, dim=1)

        attention = F.dropout(attention, training=self.training)
        h_pass = torch.matmul(attention, h)

        output_data = h_pass


        output_data = F.leaky_relu(output_data,negative_slope=self.alpha)
        output_data = F.normalize(output_data,p=2,dim=1)


        if self.bias is not None:
            output_data = output_data + self.bias
        ############## 注意这里的输出维度
        return output_data













