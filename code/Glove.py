"""
Glove:
既使用了语料库的全局统计（overall statistics）特征
也使用了局部的上下文特征（即滑动窗口）。
步骤：
1. 统计共现矩阵 X (X_ij表示单词i和单词j出现在一个窗口中的次数)
2. 训练词向量(不用神经网络)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter

# 设置GPU
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    gpus=[0]
    torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


flatten = lambda l:[item for sublist in l for item in sublist]

# 产生批量数据
def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

# 准备数据
def prepare_sequence(seq,word2index):
    idxs = list(map(lambda w:word2index[w] if word2index.get(w) is not None else word2index["<UNK>"],seq))
    return Variable(LongTensor(idxs))

def prepare_word(word,word2index):
    return Variable(LongTensor([word2index[word]]) if word2index.get(word) is not None else LongTensor([word2index["<UNK>"]]))

# 加载数据和预处理
corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:500]
corpus = [[word.lower() for word in sent] for sent in corpus]

# 建立词典
vocab = list(set(flatten(corpus)))
word2index = {}
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)

index2word = {v:k for k,v in enumerate(word2index)}

WINDOW_SIZE = 5 # 窗口大小

windows = flatten([list(nltk.ngrams(['<DUMMY>']*WINDOW_SIZE + c + ['<DUMMY>']*WINDOW_SIZE,WINDOW_SIZE*2+1)) for c in corpus])

# 窗口内的词对，用以构造共现矩阵
window_data = []
for window in windows:
    for i in range(WINDOW_SIZE*2 + 1):
        if i == WINDOW_SIZE or window[i] == '<DUMMY>':
            continue
        window_data.append((window[WINDOW_SIZE],window[i]))



# 定义权值函数
def weighting(w_i,w_j):
    try:
        x_ij = X_ik[(w_i,w_j)]
    except:
        x_ij = 1

    x_max = 100
    alpha = 0.75

    if x_ij < x_max:
        result = (x_ij/x_max)**alpha
    else:
        result = 1

    return result

# 建立词共现矩阵
X_i = Counter(flatten(corpus))
X_ik_window_5 = Counter(window_data) # 窗口大小为5的共现矩阵

X_ik = {}
weighting_dic = {}

from itertools import combinations_with_replacement

# 计算权值矩阵，加速查询
for bigram in combinations_with_replacement(vocab,2):
    if X_ik_window_5.get(bigram) is not None: # 不是非空元素
        co_occer = X_ik_window_5[bigram] # 共现次数
        X_ik[bigram] = co_occer + 1 # +· 是作平滑
        X_ik[(bigram[1],bigram[0])] = co_occer + 1 #对称矩阵
    else:
        pass

    weighting_dic[bigram] = weighting(bigram[0],bigram[1])
    weighting_dic[bigram[1],bigram[0]] = weighting(bigram[1],bigram[0])

#准备训练数据
u_p = [] #中心词
v_p = [] #上下文词
co_p = [] # log(x_ij)
weight_p = [] #f(x_ij)

for pair in window_data:
    u_p.append(prepare_word(pair[0],word2index).view(1,-1))
    v_p.append(prepare_word(pair[1],word2index).view(1,-1))

    try:
        cooc = X_ik[pair]
    except:
        cooc = 1

    co_p.append(torch.log(Variable(FloatTensor([cooc]))).view(1,-1))
    weight_p.append(Variable(FloatTensor([weighting_dic[pair]])).view(1,-1))

train_data = list(zip(u_p,v_p,co_p,weight_p))


# 定义模型
class Glove(nn.Module):
    def __init__(self,vocab_size,projection_dim):
        super(Glove, self).__init__()
        #中心词
        self.embedding_v = nn.Embedding(vocab_size,projection_dim)
        #上下文单词
        self.embedding_u = nn.Embedding(vocab_size,projection_dim)

        self.v_bias = nn.Embedding(vocab_size,1)
        self.u_bias = nn.Embedding(vocab_size,1)

        # 初始化权重
        initrange = (2.0 / (vocab_size + projection_dim)) ** 0.5  # Xavier init
        self.embedding_v.weight.data.uniform_(-initrange, initrange)  # init
        self.embedding_u.weight.data.uniform_(-initrange, initrange)  # init
        self.v_bias.weight.data.uniform_(-initrange, initrange)  # init
        self.u_bias.weight.data.uniform_(-initrange, initrange)  # init

    def forward(self, center_words,target_words,coocs,weights):
        center_embeds = self.embedding_v(center_words) #batch_size x 1 x D
        target_embeds = self.embedding_u(target_words) # batch_size x 1 X D

        center_bias = self.v_bias(center_words).squeeze(1)
        target_bias = self.u_bias(target_words).squeeze(1)

        inner_product = target_embeds.bmm(center_embeds.transpose(1,2)).squeeze(2) # batch_size x 1
        loss = weights*torch.pow(inner_product + center_bias + target_bias-coocs,2)

        return torch.sum(loss)

    def prediction(self,inputs):
        v_embeds = self.embedding_v(inputs) # batch_size x 1 x D
        u_embeds = self.embedding_u(inputs)

        return v_embeds+u_embeds


#训练
EMBEDDING_SIZE = 50
BATCH_SIZE = 256
EPOCH = 50

losses = []
model = Glove(len(word2index),EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(),lr=0.001)

for epoch in range(EPOCH):
    for i,batch in enumerate(getBatch(BATCH_SIZE,train_data)):
        inputs,targets,coocs,weights=zip(*batch)
        inputs = torch.cat(inputs) #batch_size x 1
        targets = torch.cat(targets)
        coocs = torch.cat(coocs)
        weights = torch.cat(weights)

        model.zero_grad()
        loss = model(inputs,targets,coocs,weights)

        loss.backward()
        optimizer.step()

        losses.append(loss.data.tolist())
    if epoch % 10 == 0:
        print("Epoch : %d, mean_loss : %.02f" % (epoch, np.mean(losses)))
        losses = []

#测试，计算相似度
def word_similarity(target, vocab):
    if USE_CUDA:
        target_V = model.prediction(prepare_word(target, word2index))
    else:
        target_V = model.prediction(prepare_word(target, word2index))
    similarities = []
    for i in range(len(vocab)):
        if vocab[i] == target:
            continue

        if USE_CUDA:
            vector = model.prediction(prepare_word(list(vocab)[i], word2index))
        else:
            vector = model.prediction(prepare_word(list(vocab)[i], word2index))

        cosine_sim = F.cosine_similarity(target_V, vector).data.tolist()[0]
        similarities.append([vocab[i], cosine_sim])
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10]

test = random.choice(list(vocab))
print(test)
print(word_similarity(test, vocab))