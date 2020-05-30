import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

dtype = torch.FloatTensor
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is
# short than time steps
sentences = [['ich mochte ein bier', 'i want a beer']]

# 构建源输入词典
src_vocab = []
tgt_vocab = []
for sent in sentences:
    src_vocab.extend(sent[0].split())
    tgt_vocab.extend(sent[1].split())

src_vocab_dict = {w: i + 1 for i, w in enumerate(src_vocab)}
src_vocab_dict['P'] = 0
src_vocab_size = len(src_vocab_dict)

tgt_vocab_dict = {w: i + 1 for i, w in enumerate(tgt_vocab)}
tgt_vocab_dict['P'] = 0
tgt_vocab_dict['S'] = len(tgt_vocab_dict)
tgt_vocab_dict['E'] = len(tgt_vocab_dict)
number_dict = dict(zip(tgt_vocab_dict.values(),tgt_vocab_dict.keys()))
tgt_vocab_size = len(tgt_vocab_dict)

src_len = 5
tgt_len = 5

# 超参数
d_model = 512  # 嵌入词向量维度
d_ff = 2048  # 前馈层的维度
d_k = d_v = 64  # attention中K,Q,V的维度
n_layers = 6  # 编码器/解码器的层数
n_heads = 8  # 多头自注意力机制的头数


def make_batch(sentences):
    input_batch = []
    output_batch = []
    target_batch = []
    for sen in sentences:
        input = [src_vocab_dict[n] for n in sen[0].split()] + \
            [src_vocab_dict['P']] * (src_len - len(sen[0].split()))
        output = [tgt_vocab_dict['S']] * (tgt_len - len(sen[1].split())) + [
            tgt_vocab_dict[n] for n in sen[1].split()]
        target = [tgt_vocab_dict[n] for n in sen[1].split(
        )] + [tgt_vocab_dict['E']] * (tgt_len - len(sen[1].split()))

        input_batch.append(input)
        output_batch.append(output)
        target_batch.append(target)

    return Variable(torch.LongTensor(input_batch)), \
           Variable(torch.LongTensor(output_batch)),\
           Variable(torch.LongTensor(target_batch))


def get_sinusoid_encoding_table(n_position, d_model):
    """
    位置编码计算公式
    :param n_position:单词的位置
    :param d_model: 词向量的维度
    :return: 位置编码
    """
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数维度
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数维度

    return torch.FloatTensor(sinusoid_table)

#TODO
def get_attn_pad_mask(seq_q, seq_k):
    # print(seq_q)
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

#TODO
def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self,Q,K,V,attn_mask):
        """
        缩放点积注意力机制
        :param Q: query [batch_size,n_heads,len_q,d_k(=64)]
        :param K: key [batch_size,n_heads,len_k,d_k]
        :param V: value [batch_size,n_heads,len_k,d_k]
        :param attn_mask:
        :return:
        """
        # 先得到Q和V的缩放点积
        #scores [batch_size,n_heads,len_q,len_k]
        scores = torch.matmul(Q,K.transpose(-1,-2)) / np.sqrt(d_k)
        # 进行屏蔽 attention_mask
        scores.masked_fill(attn_mask,-1e9) #将需要 mask的位置填充为负无穷
        #mask之后再做softmax,得到attention权重
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,V) #context:[batch_size,n_heads,len_q,d_k]
        return context,attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model,d_k* n_heads)
        self.W_K = nn.Linear(d_model,d_k * n_heads)
        self.W_V = nn.Linear(d_model,d_v*n_heads)

    def forward(self, Q,K,V,attn_mask):
        """
        多头注意力机制
        :param Q: [batch_size,len_q,d_model],即句子的编码
        :param K: [batch_size,len_k,d_model]
        :param V: [batch_size,len_v,d_model]
        :param attn_mask:
        :return:
        """
        residual ,batch_size = Q, Q.size(0)
        #[batch_size,len,d_model]->[batch_size,len,d_k * n_heads] ->[batch_size,len,n_heads,d_k]->[batch_size,n_heads,len,d_k]
        q_s = self.W_Q(Q).view(batch_size,-1,n_heads,d_k).transpose(1,2) #[batch_size,n_heads,len_q,d_k]
        k_s = self.W_K(K).view(batch_size,-1,n_heads,d_k).transpose(1,2) #[batch_size,n_heads,len_k,d_k]
        v_s = self.W_V(V).view(batch_size,-1,n_heads,d_k).transpose(1,2) #[batch_size,n_heads,len_k,d_k]

        attn_mask = attn_mask.unsqueeze(1).repeat(1,n_heads,1,1) #[batch_size,len_q,len_k]

        # context: [batch_size , n_heads , len_q , d_v]
        # attn: [batch_size ,n_heads , len_q(=len_k) , len_k(=len_q)]
        context,attn = ScaledDotProductAttention()(q_s,k_s,v_s,attn_mask)
        #TODO
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size , len_q , n_heads * d_v]
        output = nn.Linear(n_heads*d_v,d_model)(context)
        # output :[batch_size,len_q,d_model]
        return nn.LayerNorm(d_model)(output+residual),attn

# 每个Encoder层最后的全连接前馈层
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model,out_channels=d_ff,kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff,out_channels=d_model,kernel_size=1)

    def forward(self,inputs):
        residual = inputs #inputs:[batch_size,len_q,d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1,2)))
        output = self.conv2(output).transpose(1,2)
        return nn.LayerNorm(d_model)(output+residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs,enc_self_attn_mask):
        enc_outputs,attn =self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return  enc_outputs,attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask):
        dec_outputs,dec_self_attn = self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
        dec_outputs,dec_enc_attn = self.dec_enc_attn(dec_outputs,enc_outputs,enc_outputs,dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs,dec_self_attn,dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.scr_emb = nn.Embedding(src_vocab_size,d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self,enc_inputs): #enc_inputs:[batch_size,src_len]
        enc_outputs = self.scr_emb(enc_inputs) +  self.pos_emb(enc_inputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs,enc_self_attn = layer(enc_outputs,enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs,enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size,d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1,d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def  forward(self,dec_inputs,enc_inputs,enc_outputs):#dec_inputs:[batch_size,tgt_len]
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(dec_inputs)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs,dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        #TODO
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs,enc_inputs)

        dec_self_attns,dec_enc_attns= [],[]
        for layer in self.layers:
            dec_outputs,dec_self_attn,dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs,dec_self_attns,dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model,tgt_vocab_size,bias = False)

    def forward(self,enc_inputs,dec_inputs):
        enc_outputs,enc_self_attns = self.encoder(enc_inputs)
        dec_outputs,dec_self_attns,dec_enc_attns = self.decoder(dec_inputs,enc_inputs,enc_outputs)
        dec_logits = self.projection(dec_outputs) #dec_logits:[batch_size,src_vocab_size,tgt_vocab_size]
        return dec_logits.view(-1,dec_logits.size(-1)),enc_self_attns,dec_self_attns,dec_enc_attns


def greedy_decoder(model,enc_input,start_symbol):
    """
    贪婪编码,贪婪编码是Beam search在k=1的特殊情况，当不知道目标输入时，贪婪编码是十分必要的
    因此可以逐字生成目标输入，然后在传入transformer模型中
    :param model: Transformer模型
    :param enc_input: 编码器输入
    :param start_symbol: 序列开始标志，即'S'
    :return: 返回目标输入
    """
    enc_outputs,enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1,5).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0,5):
        dec_input[0][i] = next_symbol
        dec_outputs,_,_ = model.decoder(dec_input,enc_input,enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1,keepdim = False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input

def  showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads,n_heads))
    ax = fig.add_subplot(1,1,1)
    ax.matshow(attn,cmap = 'viridis')
    ax.set_xticklabels([''] + (sentences[0][0]+' P').split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + (sentences[0][1]+' E').split(), fontdict={'fontsize': 14})
    plt.show()


model = Transformer()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    optimizer.zero_grad()
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    loss = criterion(outputs, target_batch.contiguous().view(-1))
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

# Test
greedy_dec_input = greedy_decoder(model, enc_inputs, start_symbol=tgt_vocab_dict["S"])
predict, _, _, _ = model(enc_inputs, greedy_dec_input)
predict = predict.data.max(1, keepdim=True)[1]
print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

print('first head of last state enc_self_attns')
showgraph(enc_self_attns)

print('first head of last state dec_self_attns')
showgraph(dec_self_attns)

print('first head of last state dec_enc_attns')
showgraph(dec_enc_attns)