import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
from torch.nn import CosineSimilarity
import math

class GTransformer(nn.Module):
    def __init__(self,input_dim,hidden1_dim,hidden2_dim,hidden3_dim,output_dim,num_head1,num_head2,
                 alpha,device,type,reduction):
        super(GTransformer, self).__init__()
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
        #self.batch_size = 512
        self.timesteps = 3
        self.embed_dim = 16
        self.num_heads = 2

        #self.units = 128 #
        self.units = 64 #


        #self.s_attention = SelfAttention(self.embed_dim, self.num_heads)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.self_attention_layer1 = SelfAttention(self.embed_dim, self.num_heads).to(device)
        self.self_attention_layer2 = SelfAttention(self.embed_dim, self.num_heads).to(device)
        self.self_attention_layer3 = SelfAttention(self.embed_dim, self.num_heads).to(device)
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.weighted_fc = nn.Linear(self.embed_dim * 2, 1)  #

        #self.w_rs = nn.Parameter(torch.tensor(0.1))
        self.init_value = 0.01
        #self.w_rs = nn.Parameter(torch.ones(1, 1, 32) * self.init_value)
        #self.w_rs = nn.Parameter(torch.tensor([self.init_value]))
        self.w = nn.Parameter(0.01*torch.ones(2))

        #self.position = PositionLearnable(32,4)
        #self.multihead_attn = nn.MultiheadAttention(self.embed_dim,self.num_heads)

        #self.tansput = nn.Linear(self.units * self.timesteps, 1)
        self.tansput = nn.Linear(32 * self.timesteps, 1)  ##
        #self.tansput = nn.Linear(64, 1)
        self.fc1 = nn.Linear(self.embed_dim, self.units)
        self.fc2 = nn.Linear(self.units, self.units)


        if self.reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
        elif self.reduction == 'concate':
            self.hidden1_dim = num_head1*hidden1_dim
            self.hidden2_dim = num_head2*hidden2_dim


        self.ConvLayer1 = [AttentionLayer(input_dim,hidden1_dim,alpha) for _ in range(num_head1)]
        for i, attention in enumerate(self.ConvLayer1):
            ## ConvLayer1_AttentionHead1，ConvLayer1_AttentionHead2，ConvLayer1_AttentionHead3
            self.add_module('ConvLayer1_AttentionHead{}'.format(i),attention)
        ## gat2
        self.ConvLayer2 = [AttentionLayer(self.hidden1_dim,hidden2_dim,alpha) for _ in range(num_head2)]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module('ConvLayer2_AttentionHead{}'.format(i),attention)

        ##
        ## MLP1
        self.tf_linear1 = nn.Linear(hidden2_dim,hidden3_dim)
        self.target_linear1 = nn.Linear(hidden2_dim,hidden3_dim)
        ## MLP2
        self.tf_linear2 = nn.Linear(hidden3_dim,output_dim)
        self.target_linear2 = nn.Linear(hidden3_dim, output_dim)

        ##
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

        # # mEc2
        # self.nomal_linear1 = nn.Linear(603, input_dim)
        # self.nomal_linear2 = nn.Linear(468, input_dim)

        # mEc3（best）
        # self.nomal_linear1 = nn.Linear(495, input_dim)
        # self.nomal_linear2 = nn.Linear(335, input_dim)
        # self.nomal_linear3 = nn.Linear(241, input_dim)


        # # mEc4
        # self.nomal_linear1 = nn.Linear(242, input_dim)
        # self.nomal_linear2 = nn.Linear(311, input_dim)
        # self.nomal_linear3 = nn.Linear(285, input_dim)
        # self.nomal_linear4 = nn.Linear(233, input_dim)

        # # mEc5
        # self.nomal_linear1 = nn.Linear(233, input_dim)
        # self.nomal_linear2 = nn.Linear(300, input_dim)
        # self.nomal_linear3 = nn.Linear(242, input_dim)
        # self.nomal_linear4 = nn.Linear(87, input_dim)
        # self.nomal_linear5 = nn.Linear(209, input_dim)
        # self.nomal_linear_one = nn.Linear(1071, input_dim)

        # # mEc6
        # self.nomal_linear1 = nn.Linear(133, input_dim)
        # self.nomal_linear2 = nn.Linear(177, input_dim)
        # self.nomal_linear3 = nn.Linear(240, input_dim)
        # self.nomal_linear4 = nn.Linear(229, input_dim)
        # self.nomal_linear5 = nn.Linear(85, input_dim)
        # self.nomal_linear6 = nn.Linear(207, input_dim)

        # # mEc7
        # self.nomal_linear1 = nn.Linear(121, input_dim)
        # self.nomal_linear2 = nn.Linear(174, input_dim)
        # self.nomal_linear3 = nn.Linear(252, input_dim)
        # self.nomal_linear4 = nn.Linear(228, input_dim)
        # self.nomal_linear5 = nn.Linear(81, input_dim)
        # self.nomal_linear6 = nn.Linear(82, input_dim)
        # self.nomal_linear7 = nn.Linear(133, input_dim)

        # # mEc8
        # self.nomal_linear1 = nn.Linear(91, input_dim)
        # self.nomal_linear2 = nn.Linear(152, input_dim)
        # self.nomal_linear3 = nn.Linear(188, input_dim)
        # self.nomal_linear4 = nn.Linear(134, input_dim)
        # self.nomal_linear5 = nn.Linear(210, input_dim)
        # self.nomal_linear6 = nn.Linear(81, input_dim)
        # self.nomal_linear7 = nn.Linear(88, input_dim)
        # self.nomal_linear8 = nn.Linear(127, input_dim)

        # # mGMc2
        # self.nomal_linear1 = nn.Linear(464, input_dim)
        # self.nomal_linear2 = nn.Linear(425, input_dim)


        # # mGMc3（best）
        # self.nomal_linear1 = nn.Linear(279, input_dim)
        # self.nomal_linear2 = nn.Linear(321, input_dim)
        # self.nomal_linear3 = nn.Linear(289, input_dim)
        #self.nomal_linear_one = nn.Linear(889, input_dim)

        # # mGMc4
        # self.nomal_linear1 = nn.Linear(191, input_dim)
        # self.nomal_linear2 = nn.Linear(199, input_dim)
        # self.nomal_linear3 = nn.Linear(235, input_dim)
        # self.nomal_linear4 = nn.Linear(264, input_dim)

        # # mGMc5
        # self.nomal_linear1 = nn.Linear(121, input_dim)
        # self.nomal_linear2 = nn.Linear(169, input_dim)
        # self.nomal_linear3 = nn.Linear(227, input_dim)
        # self.nomal_linear4 = nn.Linear(177, input_dim)
        # self.nomal_linear5 = nn.Linear(195, input_dim)

        # # mGMc6
        # self.nomal_linear1 = nn.Linear(85, input_dim)
        # self.nomal_linear2 = nn.Linear(149, input_dim)
        # self.nomal_linear3 = nn.Linear(146, input_dim)
        # self.nomal_linear4 = nn.Linear(186, input_dim)
        # self.nomal_linear5 = nn.Linear(138, input_dim)
        # self.nomal_linear6 = nn.Linear(185, input_dim)

        # # mGMc7
        # self.nomal_linear1 = nn.Linear(63, input_dim)
        # self.nomal_linear2 = nn.Linear(115, input_dim)
        # self.nomal_linear3 = nn.Linear(112, input_dim)
        # self.nomal_linear4 = nn.Linear(173, input_dim)
        # self.nomal_linear5 = nn.Linear(137, input_dim)
        # self.nomal_linear6 = nn.Linear(118, input_dim)
        # self.nomal_linear7 = nn.Linear(171, input_dim)

        # # mGMc8
        # self.nomal_linear1 = nn.Linear(54, input_dim)
        # self.nomal_linear2 = nn.Linear(109, input_dim)
        # self.nomal_linear3 = nn.Linear(111, input_dim)
        # self.nomal_linear4 = nn.Linear(105, input_dim)
        # self.nomal_linear5 = nn.Linear(128, input_dim)
        # self.nomal_linear6 = nn.Linear(108, input_dim)
        # self.nomal_linear7 = nn.Linear(109, input_dim)
        # self.nomal_linear8 = nn.Linear(165, input_dim)

        # # mLc2 （best）
        # self.nomal_linear1 = nn.Linear(425, input_dim)
        # self.nomal_linear2 = nn.Linear(422, input_dim)
        #self.nomal_linear_one = nn.Linear(847, input_dim)

        # # mLc3
        # self.nomal_linear1 = nn.Linear(276, input_dim)
        # self.nomal_linear2 = nn.Linear(280, input_dim)
        # self.nomal_linear3 = nn.Linear(291, input_dim)

        # # mLc4
        # self.nomal_linear1 = nn.Linear(204, input_dim)
        # self.nomal_linear2 = nn.Linear(229, input_dim)
        # self.nomal_linear3 = nn.Linear(215, input_dim)
        # self.nomal_linear4 = nn.Linear(199, input_dim)

        # # mLc5
        # self.nomal_linear1 = nn.Linear(159, input_dim)
        # self.nomal_linear2 = nn.Linear(172, input_dim)
        # self.nomal_linear3 = nn.Linear(160, input_dim)
        # self.nomal_linear4 = nn.Linear(196, input_dim)
        # self.nomal_linear5 = nn.Linear(160, input_dim)

        # # mLc6
        # self.nomal_linear1 = nn.Linear(119, input_dim)
        # self.nomal_linear2 = nn.Linear(158, input_dim)
        # self.nomal_linear3 = nn.Linear(160, input_dim)
        # self.nomal_linear4 = nn.Linear(184, input_dim)
        # self.nomal_linear5 = nn.Linear(165, input_dim)
        # self.nomal_linear6 = nn.Linear(61, input_dim)

        # # mLc7
        # self.nomal_linear1 = nn.Linear(80, input_dim)
        # self.nomal_linear2 = nn.Linear(135, input_dim)
        # self.nomal_linear3 = nn.Linear(138, input_dim)
        # self.nomal_linear4 = nn.Linear(131, input_dim)
        # self.nomal_linear5 = nn.Linear(152, input_dim)
        # self.nomal_linear6 = nn.Linear(156, input_dim)
        # self.nomal_linear7 = nn.Linear(55, input_dim)
        #self.nomal_linear = nn.Linear(847, input_dim)

        # # mLc8
        # self.nomal_linear1 = nn.Linear(63, input_dim)
        # self.nomal_linear2 = nn.Linear(114, input_dim)
        # self.nomal_linear3 = nn.Linear(113, input_dim)
        # self.nomal_linear4 = nn.Linear(147, input_dim)
        # self.nomal_linear5 = nn.Linear(161, input_dim)
        # self.nomal_linear6 = nn.Linear(153, input_dim)
        # self.nomal_linear7 = nn.Linear(80, input_dim)
        # self.nomal_linear8 = nn.Linear(16, input_dim)

        # mDC
        self.nomal_linear1 = nn.Linear(137, input_dim)
        self.nomal_linear2 = nn.Linear(177, input_dim)
        self.nomal_linear3 = nn.Linear(69, input_dim)

        # hHEP
        # self.nomal_linear1 = nn.Linear(81, input_dim)
        # self.nomal_linear2 = nn.Linear(70, input_dim)
        # self.nomal_linear3 = nn.Linear(116, input_dim)
        # self.nomal_linear4 = nn.Linear(91, input_dim)
        # self.nomal_linear5 = nn.Linear(67, input_dim)

        #self.nomal_linear1 = nn.Linear(425, input_dim)

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


        ##
        if self.type == 'MLP':
            self.linear = nn.Linear(2*output_dim, 2) ##

        ## lstm
        # self.rnn = nn.LSTM(  #
        #     input_size=32,  #
        #     hidden_size=256,  # rnn hidden unit
        #     num_layers=2,  #
        #     batch_first=True,  #
        # )

        ## gru
        self.gru = nn.GRU(input_size=32, hidden_size=128, num_layers=2, batch_first=True)


        self.attention = Attention(self.units) #

        #self.tf_ecoder = TransformerEcoder()
        #self.mutil_attention = MultiHeadAttentionLayer(inputs)

        #self.out = nn.Linear(128, 1)  #
        # self.out = nn.Linear(128,10)
        # self.fout = nn.Linear(10,1)

        #self.out = nn.Linear(128, 16)
        self.final_out = nn.Linear(self.units,16)
        self.final_out_causal = nn.Linear(2*self.units,3)
        self.final_out_causal_two_label = nn.Linear(2 * self.units, 1)


        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters() ## attentionlayer

        for attention in self.ConvLayer2:
            attention.reset_parameters()

        nn.init.xavier_uniform_(self.tf_linear1.weight,gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)




    def encode(self,x,adj):
        ##
        if self.reduction =='concate':
            x = torch.cat([att(x, adj) for att in self.ConvLayer1], dim=1) #
            x = F.elu(x)

        elif self.reduction =='mean':
            x = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer1]), dim=0) #
            x = F.elu(x)

        else:
            raise TypeError


        with torch.no_grad():
            out = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer2]),dim=0) #

        return out


    def decode(self,tf_embed,target_embed):

        # if self.type =='dot':
        #     ## mul()
        #     prob = torch.mul(tf_embed, target_embed)
        #     prob = torch.sum(prob,dim=1).view(-1,1) ##
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
        ##
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        #r_out, (h_n, h_c) = self.gru(h_x, None)  #


        #inputs = torch.randn(batch_size, timesteps, embed_dim)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        inputs_tf = h_x_tf  # (512,time_point,32)
        inputs_target = h_x_target
        #
        # pos_encoding = np.zeros((1, t_p, self.embed_dim)) #
        # #pos_encoding = torch.tensor(pos_encoding)
        # #pos_encoding = pos_encoding.to(device)
        # for i in range(t_p):
        #     for j in range(self.embed_dim):
        #         pos_encoding[0, i, j] = np.sin(i / (10000 ** ((2 * j) / self.embed_dim)))
        # pos_encoding = torch.tensor(pos_encoding, dtype=torch.float32)
        # pos_encoding = pos_encoding.to(device)

        ##
        pos_encoding = torch.zeros((t_p, self.embed_dim))
        pos = torch.arange(0, t_p, dtype=torch.float).unsqueeze(1)
        #div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim))  ##
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim))
        pos_encoding[:, 0::2] = torch.sin(pos * div_term)
        pos_encoding[:, 1::2] = torch.cos(pos * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        pos_encoding = pos_encoding.to(device)


        x_pos_tf = inputs_tf + pos_encoding  ##
        x_pos_target = inputs_target + pos_encoding
       # x = inputs
        #x = self.position(inputs)  ##
        x_tf = x_pos_tf.to(device)
        x_target = x_pos_target.to(device)
        # 应用多头注意力机制层
        #multihead = nn.MultiheadAttention(self.embed_dim,self.num_heads).to(device)
        #attn_output, attn_weights = self.multihead_attn(x, x, x)
        #attn_output, attn_weights = multihead(x,x,x)

        ##
        # attention = MultiHeadAttentionLayer(self.embed_dim, self.num_heads).to(device)
        # x = attention(x)

        ##
        #input = inputs
        x_tf = self.self_attention_layer1(x_tf)
        x_target = self.self_attention_layer1(x_target)
        # x_tf = self.self_attention_layer2(x_tf)
        # x_target = self.self_attention_layer2(x_target)
        # x_tf = self.self_attention_layer3(x_tf)
        # x_target = self.self_attention_layer3(x_target)
        # #x = self.self_attention_layer2(x)
        #x = self.self_attention_layer3(x)
        #output = 16 * input + x
        ##
        #output = 8*x_pos + 1*x

        ##
        #
        # c = torch.cat((x_pos, x), dim=2)
        # #
        # w = torch.sigmoid(self.weighted_fc(c))
        # #
        # output = w * x_pos + (1 - w) * x

        ##
        #w = self.w_rs.expand(-1, 32)
        #w = self.w_rs.expand(x_pos.size())
        #w = self.w_rs
        w_rs = 100*self.w
        w1 = torch.exp(w_rs[0]) / torch.sum(torch.exp(w_rs))
        w2 = torch.exp(w_rs[1]) / torch.sum(torch.exp(w_rs))

        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))

        #print("w：{}".format(w_rs))
        #w_ = w*800
        #print("w_:{}".format(w_))
        #
        #output = w1 * x_pos + w2 * x
        #output = 0 * x_pos + 1 * x
        output_tf = w1 * inputs_tf + w2 * x_tf
        output_target = w1 * inputs_target + w2 * x_target
        # output = self.layer_norm(inputs + output)
        x_tf = self.layer_norm(output_tf)
        x_target = self.layer_norm(output_target)

        # prob = torch.mul(x_tf[:,-1,:], x_target[:,-1,:])
        # prob = torch.sum(prob,dim=1).view(-1,1) ##
        # return prob

        ## transformerecoder
        #x = self.tf_ecoder(x)

        #
        x_tf = F.relu(self.fc1(x_tf))         ##
        x_target = F.relu(self.fc1(x_target))


        #x = F.relu(self.fc2(x)) ## (512,4,64)
        #print(x.size())
        #
        #x = x.view(x.size()[0], -1) ## (512,64*4)
        # print(x)
        # print(x.size())
        # print(x)
        # print(x.size())
        #
        #output_layer = nn.Linear(self.units * self.timesteps, 1)
        #output = self.tansput(x)
        # print(output)
        # print(output.size())



        ##
        #attention_output = self.attention(r_out)

        attention_output_tf = self.attention(x_tf)  ##
        attention_output_target = self.attention(x_target)



        #print(attention_output_tf.size())


        ##
        if self.flag:
            liner_concat = torch.cat((attention_output_tf,attention_output_target),dim=1)
            #print(liner_concat.size())
            prob = self.final_out_causal(liner_concat)
            return prob

        else:
            # linear_output_tf = self.final_out(attention_output_tf)
            # linear_output_target = self.final_out(attention_output_target)
            #
            # ##
            # self.tf_ouput = linear_output_tf
            # self.target_output = linear_output_target
            #
            # prob = torch.mul(linear_output_tf, linear_output_target)
            # prob = torch.sum(prob,dim=1).view(-1,1) ##

            ##
            # prob = torch.mul(attention_output_tf, attention_output_target)
            # prob = torch.sum(prob, dim=1).view(-1, 1)  ##整个batch的预测结果，是一列tensor


            ########### two_label_causal
            liner_concat = torch.cat((attention_output_tf, attention_output_target), dim=1)
            prob = self.final_out_causal_two_label(liner_concat)

            return prob,attention_output_tf,attention_output_target



        #
        # torch.Size([64, 28, 128])->torch.Size([64,128])
        #out = self.out(r_out[:,-1, :])  # torch.Size([64, 128])-> torch.Size([64, 10])
        #out = F.relu(out)
        #out = F.dropout(out,p=0.5)
        #out = self.final_out(out)
        #out = F.leaky_relu(out)

        ##
        # l_out = []
        # for i in range(t_p):
        #     out = self.out(r_out[:, i, :])
        #     out = F.relu(out)
        #     l_out.append(out)
        # ll_out = torch.cat(l_out, dim=1)
        # #ll_out = self.out(ll_out)
        # #F.dropout(ll_out,p=0.01)
        # lll_out = self.final_out(ll_out)

        ##
        # l_out = []
        # for i in range(t_p):
        #     out = self.out(r_out[:, i, :])
        #     #out = F.relu(out)
        #     #out = out.cpu().detach().numpy()
        #     l_out.append(out)
        #
        # feature_vectors = l_out
        # #
        # features_matrix = torch.stack(feature_vectors,dim=1)
        # pooled_features = torch.mean(features_matrix,dim=1)
        #
        # lll_out = self.final_out(pooled_features)
        #return prob

    ##
    def forward(self,x,adj,train_sample):
        ##
        train_tf_total = []
        train_target_total = []
        for i in range(self.time_point):
            if i == 0:
                e = self.nomal_linear1(x[i])
            elif i == 1:
                e = self.nomal_linear2(x[i])
            elif i == 2:
                e = self.nomal_linear3(x[i])
            elif i == 3:
                e = self.nomal_linear4(x[i])
            elif i == 4:
                e = self.nomal_linear5(x[i])
            # elif i == 5:
            #     e = self.nomal_linear6(x[i])
            # elif i == 6:
            #     e = self.nomal_linear7(x[i])
            # elif i == 7:
            #     e = self.nomal_linear8(x[i])
            # elif i == 8:
            #     e = self.nomal_linear9(x[i])
            else:
                e = self.nomal_linear6(x[i])
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

            ##
            # self.tf_ouput = tf_embed
            # self.target_output = target_embed

            ##
            train_tf = tf_embed[train_sample[:,0]] #
            train_target = target_embed[train_sample[:, 1]] #
            train_tf_total.append(train_tf)
            train_target_total.append(train_target)
        ##
        pred = self.decode(train_tf_total, train_target_total)

        return pred

    def get_embedding(self):
        return self.tf_ouput, self.target_output



class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        #
        self.query_weight = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.key_weight = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.value_weight = nn.Linear(d_model, d_model * n_heads, bias=False)

        #
        self.output_weight = nn.Linear(d_model * n_heads, d_model)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]

        batch_size, seq_len, d_model = x.size()

        #
        query = self.query_weight(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        key = self.key_weight(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        value = self.value_weight(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_v]

        #
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))  # [batch_size, n_heads, seq_len, seq_len]

        #
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        #
        attn_weights = F.softmax(attn_scores, dim=-1)

        #
        attn_output = torch.matmul(attn_weights, value)  # [batch_size, n_heads, seq_len, d_v]

        #
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # [batch_size, seq_len, n_heads * d_v]

        #
        output = self.output_weight(attn_output)  # [batch_size, seq_len, d_model]

        return output




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
        query = self.query_fc(inputs).view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2) ##
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
        #
        linear_output = self.linear(inputs)
        #
        attention_weights = self.softmax(linear_output)
        #
        attention_output = torch.sum(attention_weights * inputs, dim=1)
        return attention_output


class AttentionLayer(nn.Module):
    def __init__(self,input_dim,output_dim,alpha=0.2,bias=True):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim #16
        self.alpha = alpha

        ## w = (output_dim*input_dim)
        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim,self.output_dim))
        ##
        self.a = nn.Parameter(torch.zeros(size=(2*self.output_dim,1))) #

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        ##
        self.reset_parameters()


    def reset_parameters(self):
        ##
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
        ##
        h = torch.matmul(x, self.weight) # x:1120*421   weight:16*421
        e = self._prepare_attentional_mechanism_input(h) ##

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
        ##############
        return output_data













