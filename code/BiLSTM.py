import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

dtype = torch.FloatTensor

sentence = ('Lorem ipsum dolor sit amet consectetur adipisicing elit '
    'sed do eiusmod tempor incididunt ut labore et dolore magna '
    'aliqua Ut enim ad minim veniam quis nostrud exercitation'
            )

word2index = {w:i for i,w in enumerate(list(set(sentence.split())))}
index2word = {i:w for i,w in enumerate(list(set(sentence.split())))}
vocab_size = len(word2index)
max_len = len(sentence.split())
hidden_size = 5

def make_batch(sentence):
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i,word in enumerate(words[:-1]):
        input = [word2index[w] for w in words[:(i+1)]]
        input = input + [0]*(max_len-len(input))
        target = word2index[words[i+1]]

        input_batch.append(np.eye(vocab_size)[input])
        target_batch.append(target)

    return Variable(torch.Tensor(input_batch)), Variable(torch.LongTensor(target_batch))


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size = vocab_size,hidden_size=hidden_size,bidirectional=True)
        self.W = nn.Parameter(torch.randn([hidden_size*2,vocab_size]).type(dtype))
        self.b = nn.Parameter(torch.randn([vocab_size]).type(dtype))

    def forward(self,X):
        input = X.transpose(0,1)

        hidden_state = Variable(torch.zeros(1*2,len(X),hidden_size))
        cell_state = Variable(torch.zeros(1 * 2, len(X), hidden_size))

        outputs,(_,_) = self.lstm(input,(hidden_state,cell_state))
        outputs = outputs[-1]
        model = torch.mm(outputs,self.W) + self.b
        return model

input_batch,target_batch = make_batch(sentence)

model = BiLSTM()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

#训练
for epoch in range(10000):
    optimizer.zero_grad()
    output = model(input_batch)
    loss = criterion(output,target_batch)
    if (epoch+1) % 1000 == 0:
        print('Epoch:','%04d' % (epoch+1),'cost = ','{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

predict  = model(input_batch).data.max(1,keepdim=True)[1]
print(sentence)
print([index2word[n.item()] for n in predict.squeeze()])
