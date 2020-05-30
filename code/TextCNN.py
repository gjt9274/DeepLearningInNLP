import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor


# TextCNN参数
embedding_size = 2  # 词嵌入大小，论文中 k
sequence_length = 3  # 句子长度，论文中 n
num_classes = 2  # 类别数，0或1
filter_sizes = [2, 2, 2]  # 卷积核大小
num_filters = 3  # 卷积核数目，卷积输出的通道数

sentences = [
    "i love you",
    "he loves me",
    "she likes baseball",
    "i hate you",
    "sorry for that",
    "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

word_list = " ".join(sentences).split()
word_list = list(set(word_list))

word2id = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word2id)

inputs = []
for sen in sentences:
    inputs.append(np.asarray([word2id[w] for w in sen.split()]))

targets = []
for label in labels:
    targets.append(label)

input_batch = Variable(torch.LongTensor(inputs))
target_batch = Variable(torch.LongTensor(targets))

# 定义模型


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()

        # 总共的卷积核数目
        self.num_filters_total = num_filters * len(filter_sizes)
        # 嵌入矩阵
        self.W = nn.Parameter(torch.empty(
            vocab_size, embedding_size).uniform_(-1, 1)).type(dtype)
        # 最后输出层权重
        self.Weight = nn.Parameter(torch.empty(
            self.num_filters_total, num_classes).uniform_(-1, 1)).type(dtype)
        # 输出层的偏置
        self.Bias = nn.Parameter(0.1 * torch.ones([num_classes])).type(dtype)

    def forward(self, X):
        # [batch_size,sequence_length,embedding_size]
        embedded_chars = self.W[X]
        # 额外增加一个维度，即通道数 [batch_size,channel,sequence_length,embedding_size]
        embedded_chars = embedded_chars.unsqueeze(1)

        pooled_outputs = []  # 池化层输出
        for filter_size in filter_sizes:
            # conv2d:(in_channels, out_channels, kernel_size, stride=1,
            # padding=0, dilation=1, groups=1, bias=True)
            conv = nn.Conv2d(1, num_filters, (filter_size, embedding_size), bias=True)(
                embedded_chars)  # [batch_size,out_channels,out_H,out_W]

            h = F.relu(conv)  # 激活函数

            # 最大池化层
            mp = nn.MaxPool2d((sequence_length - filter_size + 1, 1))
            # permute()将维度换位，将1和3维换位
            # [batch_size,out_channels,out_H,out_W]->[batch_size,out_W,out_H,out_channels]
            pooled = mp(h).permute(0, 3, 2, 1)

            pooled_outputs.append(pooled)

        #[batch_size,out_height,out_weight,out_channels * len(filter_sizes)]
        h_pool = torch.cat(pooled_outputs, len(filter_sizes))
        # [batch_size, out_height * out_width* out_channels * len(filter_sizes]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])

        # [batch_size,num_classes]
        output = torch.mm(h_pool_flat, self.Weight) + self.Bias

        return output


model = TextCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5000):
    optimizer.zero_grad()
    outputs = model(input_batch)
    loss = criterion(outputs, target_batch)

    if (epoch + 1) % 1000 == 0:
        print('Epoch:','%04d' % (epoch + 1),'cost = ','{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

#测试
test_text = 'sorry hate you'
tests = [np.asarray([word2id[w] for w in test_text.split()])]
test_batch = Variable(torch.LongTensor(tests))

predict = model(test_batch).data.max(1,keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text,'is Bad Mean...')
else:
    print(test_text,'is Good Mean!')