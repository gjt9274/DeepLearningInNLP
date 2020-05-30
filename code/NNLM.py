"""
NNLM：
语言模型，通过前n-1个单词，来预测第n个单词
1. 输入：输入n-1个单词的one-hot编码。  # n-1 x |V|(词表大小)
2. 经过一个embedding层，通过矩阵 C(|V| x m )嵌入成 m 维向量。 # n-1 x m
3. 然后通过一个前馈神经网络，由 tanh 隐藏层和softmax的输出层组成，将n-1个词向量
映射成一个长度为 |V| 的概率分布向量
4. 从嵌入向量有时也直接到softmax层
参数:
C:嵌入矩阵， # |V| x m
H:隐藏层tanh的权值矩阵，# h x (n-1)m
d:隐藏层偏置矩阵， # h
U:隐藏层到softmax的权值矩阵，# |V| x h
b:到输出层softmax的偏置矩阵
W:词向量层到softmax层的权值矩阵，#|V| x (n-1)m
模型公式：
y = b + Wx+ Utanh(d+Hx)
损失函数：
交叉熵损失函数
优化方法：
随机梯度下降
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor

# 数据预处理
sentences = ['i love dog',
             'i like coffee',
             'i hate milk']

word_list = " ".join(sentences).split()  # split()默认是所有的空字符，包括空格，滑囊，制表符
word_list = list(set(word_list))
word2id = {w: i for i, w in enumerate(word_list)}
id2word = {i: w for i, w in enumerate(word_list)}

# NNLM 的参数设置
V = len(word_list)  # 词表大小
n_step = 2  # 文中的 n-1
h = 2  # 隐藏层维度
m = 2  # 词向量维度

# 批处理
def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word2id[w] for w in word[:-1]]  # 前 2 个单词
        target = word2id[word[-1]]  # 句子的最后一个单词

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch


# 模型
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        # 嵌入矩阵 C (V , m)
        self.C = nn.Embedding(V, m)
        # 隐藏层权值矩阵 H (h,(n-1)m)
        self.H = nn.Parameter(torch.randn(n_step * m, h).type(dtype))
        # 直接连接的权值矩阵 W (V,(n-1)m)
        self.W = nn.Parameter(torch.randn(n_step * m, V).type(dtype))
        # 偏置向量 d (h,)
        self.d = nn.Parameter(torch.randn(h).type(dtype))
        # 隐藏层到softmax的权值矩阵 U (V,h)
        self.U = nn.Parameter(torch.randn(h, V).type(dtype))
        # 偏置向量 b (V,)
        self.b = nn.Parameter(torch.randn(V).type(dtype))

    def forward(self, X):
        # 词嵌入
        X = self.C(X)
        # 将 n-1 个词向量合并成一个
        X = X.view(-1, n_step * m)  # (batch_size,(n-1)*m)
        tanh = torch.tanh(self.d + torch.mm(X, self.H))  # （batch_size,h)
        output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U)
        return output


# 设置模型，损失函数，优化方法
model = NNLM()

# 将输入进行softmax处理后，再求交叉熵损失，所以模型计算最后不需要sotmax层
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_batch, target_batch = make_batch(sentences)

# 需要将输入进行处理
input_batch = Variable(torch.LongTensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))

# 训练
for epoch in range(5000):
    # 初始化优化器
    optimizer.zero_grad()

    output = model(input_batch)

    # 求损失
    loss = criterion(output, target_batch)

    # 打印损失变化
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# 预测
# max会返回张量最大的值及其索引，[0]是值，[1]是索引
predict = model(input_batch).data.max(1,keepdim=True)[1]

print([sen.split()[:2] for sen in sentences],'->',[id2word[index.item()] for index in predict.squeeze()])
