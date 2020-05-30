"""
模型结构与NNLM类似，不过是相反的。
通过中心词来预测其上下文的单词的分布
模型：
1. 中心词 x 的one-hot编码 # |v|
2. 通过嵌入矩阵 V (n x |V|) 得到中心词的嵌入词向量v = Vx # n
3. 通过矩阵 U (|V| x n )来计算得分向量 z = Uv
4. 将得分向量通过softmax转换成概率 y = softmax(v)
优化方法 ：
随机梯度下降
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor

# 预处理数据
sentences = [ "i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequences = " ".join(sentences).split()
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word2id = {w:i for i,w in enumerate(word_list)}

# word2vec 参数
batch_size = 20
embedding_size = 2 # 嵌入层向量维度，为了方便画图展示
voc_size = len(word_list) # 词表大小

# 产生随机批量
def random_batch(data,size):
    random_inputs = []
    random_labels = []
    # 从数据data中随机选size组
    random_index = np.random.choice(range(len(data)),size,replace=False)

    for i in random_index:
        #np.eye可以用来产生one-hot编码
        random_inputs.append(np.eye(voc_size)[data[i][0]]) #中心词
        random_labels.append(data[i][1]) # 上下文单词

    return random_inputs,random_labels

# 准备窗口大小为1的skip-gram数据
skip_grams = []
for i in range(1,len(word_sequences)-1):
    target = word2id[word_sequences[i]] #中心词
    context = [word2id[word_sequences[i-1]],word2id[word_sequences[i+1]]] # 上下文单词

    for w in context:
        skip_grams.append([target,w])

# 构建word2vec模型
class SkipGram(nn.Module):
    def __init__(self):
        super(SkipGram, self).__init__()
        #嵌入矩阵
        self.V = nn.Parameter(torch.randn(voc_size,embedding_size).type(dtype))
        #得分矩阵
        self.U = nn.Parameter(torch.randn(embedding_size,voc_size).type(dtype))

    def forward(self,X):
        hidden_layer = torch.matmul(X,self.V) #(batch_size,embedding_size)
        output_layer = torch.matmul(hidden_layer,self.U)#(batch_size,voc_size)

        return output_layer

model = SkipGram()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

#训练
for epoch in range(5000):
    input_batch,target_batch = random_batch(skip_grams,batch_size)

    input_batch = Variable(torch.Tensor(input_batch))
    target_batch = Variable(torch.LongTensor(target_batch))

    optimizer.zero_grad()
    output = model(input_batch)

    loss = criterion(output,target_batch)
    if (epoch+1) %1000 == 0:
        print('Epoch:','%04d'%(epoch+1),'cost=','{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()


for i,label in enumerate(word_list):
    V,U = model.parameters()
    x,y = float(V[i][0]),float(V[i][1])
    plt.scatter(x,y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.show()