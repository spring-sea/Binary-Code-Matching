import json
import numpy as np
import torch
from torch import nn
import  torch.nn.functional as Fa
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import argparse

class cross_code_lstm(nn.Module):
    def __init__(self,vocab_size,padding_idx,pad_size):
        super().__init__()
        self.padding_idx = padding_idx
        self.code_encoder = nn.Embedding(vocab_size,32, padding_idx=padding_idx)
        #self.position_encoder = nn.Embedding(pad_size+1,64, padding_idx=padding_idx)    

        self.code_lstm  = torch.nn.LSTM(
                                input_size = 32*5,#dimension词向量维度
                                hidden_size = 32,#表示输出的特征维度，如果没有特殊变化，相当于out
                                num_layers = 2,# 表示网络的层数
                                bidirectional = True,#双向向RNN
                                batch_first = True,
                                dropout=0.2
                                )        

        self.fc0 = nn.Sequential(
            nn.Linear(32*4, 32),
        )
        
        self.fc01 = nn.Sequential(
            nn.Linear(32*2, 32),
        )     

    def cross_attention(self,lstm_output, h_t):
           # lstm_output [3, 10, 16]  h_t[10, 16]
           h_t = h_t.unsqueeze(0)
           # [10, 16, 1]
           h_t = h_t.permute(1, 2, 0)
           
           attn_weights = torch.bmm(lstm_output, h_t)
           attn_weights = attn_weights.permute(1, 0, 2).squeeze()
    
           # [3, 10]
           if len(attn_weights.shape)==2:
               attention = Fa.softmax(attn_weights, 1)
               attn_out = torch.bmm(lstm_output.transpose(1, 2), attention.unsqueeze(-1).transpose(1,0))

               
           else:
               attention = Fa.softmax(attn_weights, 0)
               attn_out = torch.matmul(lstm_output.transpose(1, 2), attention.unsqueeze(-1))
           # bmm: [10, 16, 3] [10, 3, 1]
    
           return attn_out.squeeze()

    def get_code_post_feat(self,sequece_token):
        
        lengths = sequece_token[:,:,0] !=self.padding_idx
        lengths = lengths.type(torch.IntTensor).sum(dim=1)
        
        index_code_feat = self.code_encoder(sequece_token)
        shape = index_code_feat.shape
        #index_code_feat = index_code_feat.resize(shape[0],shape[1],shape[2]*shape[3])
        
        #index_code_feat = index_code_feat.sum(dim=-2)
        
        #index_code_feat = index_code_feat[:,:,0:3,:]
        shape = index_code_feat.shape        
        index_code_feat = index_code_feat.resize(shape[0],shape[1],shape[2]*shape[3])
        
        index_sequece_feat = index_code_feat#+index_position_feat
        #mask = mask.repeat(1,1,4)
        index_sequece_feat = pack_padded_sequence(input=index_sequece_feat, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_out,(index_sequece_feat2,_) = self.code_lstm(index_sequece_feat)
        
        
        index_sequece_feat2 = torch.cat([index_sequece_feat2[i] for i in range(4)],dim=1)
        
        index_sequece_feat2 = self.fc0(index_sequece_feat2)
        index_sequece_feat2 = torch.relu(index_sequece_feat2)
        
        packed_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        packed_out = self.fc01(packed_out)        
        
        
        #自注意力
        #index_sequece_feat2 = self.cross_attention(packed_out,index_sequece_feat2) 
        
        
        return index_sequece_feat2,packed_out  


    def forward(self,position_token,index_sequece_token,postive_sequece_token,nagetive_sequece_token=None):
        
        index_sequece_feat2,index_out = self.get_code_post_feat(index_sequece_token)
        postive_sequece_feat2,postive_out = self.get_code_post_feat(postive_sequece_token) 
        
        
        #postive_sequece_feat2 = self.cross_attention(index_out,postive_sequece_feat2) 
        
        if nagetive_sequece_token!=None:
            nagetive_sequece_feat2,nagetive_out = self.get_code_post_feat(nagetive_sequece_token)        
            #nagetive_sequece_feat2 = self.cross_attention(index_out,nagetive_sequece_feat2) 
            
            
            return index_sequece_feat2,postive_sequece_feat2,nagetive_sequece_feat2
        
        return index_sequece_feat2,postive_sequece_feat2

def get_token(token,padsize):
    
    
    tokens = token.split("~~")
    tokens = [i.replace("+",",+,").replace("[","").replace("]","").split(",") for i in tokens]
    token_ids = []
    for i in tokens:
        token_id = []
        for j in i:
            if len(token_id)<=5:
                if token2id_dict.get(j,'-1')!='-1':
                    token_id = token_id+ [token2id_dict[j]]
                    
        if len(token_id)<=5:
            token_id = token_id+[len(token2id_dict)]*(5-len(token_id))
        
        if len(token_id)>5:
            token_id = token_id[:5]
        
        token_ids = token_ids+ [token_id]
    
    if len(token_ids)>padsize:
        token_ids = token_ids[:padsize]
    else:
        token_ids = token_ids+[[len(token2id_dict)]*5]*(padsize - len(token_ids))
        
    token_ids = np.array(token_ids)
    return token_ids

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='计算两端汇编代码的相似度')
    parser.add_argument("--key_code_dir", type=str,default="data/test_code.txt")
    parser.add_argument("--value_code_dir", type=str,default="data/test_code_positive.txt")
    #parser.add_argument("--out_put_dir", type=str)
    args = parser.parse_args()    
    
    
    with open("token_dict.json", "r") as f:
        token2id_dict = json.loads(f.read())
        token2id_dict['+'] = len(token2id_dict)
        token2id_dict['cs:'] = len(token2id_dict)    
    
    
    with open(args.key_code_dir,"r") as f:
        line = [i.strip() for i in f.readlines()]
        index_str = "~~".join([i.replace("    ",",") for i in line]).replace('\xa9','')

    with open(args.value_code_dir,"r") as f:
        line = [i.strip() for i in f.readlines()]
        query_str = "~~".join([i.replace("    ",",") for i in line]).replace('\xa9','')

    
    index_token = get_token(index_str,256)
    query_token = get_token(query_str,256)

        
    vocab_size = len(token2id_dict)+1
    padding_idx = len(token2id_dict)
    print(padding_idx)
    pad_size = 256    
    model = cross_code_lstm(vocab_size,padding_idx,pad_size).cuda()
    model.load_state_dict(torch.load("model_step_4w256.pt"))
    index_token = torch.from_numpy(index_token).type(torch.LongTensor).cuda()
    query_token = torch.from_numpy(query_token).type(torch.LongTensor).cuda()

    index_feat,_ = model.get_code_post_feat(index_token.unsqueeze(dim=0))
    query_feat,_ = model.get_code_post_feat(query_token.unsqueeze(dim=0))
    
    
    score = torch.cosine_similarity(index_feat,query_feat).data.cpu().numpy()[0]
    print("两段代码的相似度为：",score)
    
    
    
    