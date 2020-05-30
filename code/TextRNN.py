import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

dtype = torch.FloatTensor

sentences = ['i like dog','i love coffee','i hate milk']

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word2index = {w:i for i,w in enumerate(word_list)}
index2word = {i:w for  i,w in enumerate(word_list)}
vocab_size = len(word2index)


# TextRNN 超参数
batch_size = len(sentences)
seq_len = 2 # 步数
hidden_size = 5 # 每个隐藏节点的维度

def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word2index[w] for w in word[:-1]]
        target = word2index[word[-1]]

        input_batch.append(np.eye(vocab_size)[input]) #生成one-hot编码
        target_batch.append(target)

    return input_batch,target_batch


# 转成torch.tensor
input_batch,target_batch = make_batch(sentences)
input_batch = Variable(torch.Tensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))


#RNN模型
class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size=vocab_size,hidden_size=hidden_size)
        #最后的线性层权重
        self.W = nn.Parameter(torch.randn([hidden_size,vocab_size]).type(dtype))
        #最后的线性层偏置
        self.b = nn.Parameter(torch.randn([vocab_size]).type(dtype))

    def forward(self,hidden,X):
        X = X.transpose(0,1) #[batch_size,seq_len,vocab_size]->[seq_len,batch_size,vocab_size]
        outputs,hidden = self.rnn(X,hidden)
        #outputs:[seq_len,batch_size,num_direction(=1)*hidden_size]
        #hidden:[num_layers * num_directions, batch_size, hidden_size]

        outputs = outputs[-1] #取最后一个输出，[batch_size,num_direction(=1)*hidden_size]

        model = torch.mm(outputs,self.W) + self.b #model:[batch_size,vocab_size]
        return model



model = TextRNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

#训练
for epoch in range(5000):
    optimizer.zero_grad()

    #初始隐藏单元
    hidden = Variable(torch.zeros(1,batch_size,hidden_size))
    # input_batch:[batch_size,seq_len,vocab_size]
    output = model(hidden,input_batch)
    loss = criterion(output,target_batch)

    if (epoch+1) % 1000 == 0:
        print('Epoch:','%04d'%(epoch+1),'cost=','{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

#预测
# Predict
hidden = Variable(torch.zeros(1, batch_size, hidden_size))
predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
print([sen.split()[:2] for sen in sentences], '->', [index2word[n.item()] for n in predict.squeeze()])