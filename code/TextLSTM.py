import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

dtype = torch.FloatTensor

char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
char2index = {c: i for i, c in enumerate(char_arr)}
index2char = {i: c for i, c in enumerate(char_arr)}
vocab_size = len(char2index)

seq_data = [
    'make',
    'need',
    'coal',
    'word',
    'love',
    'hate',
    'live',
    'home',
    'hash',
    'star']

# 超参数
n_step = 3  # 步长
hidden_size = 128


def make_batch(seq_data):
    input_batch, target_batch = [], []

    for seq in seq_data:
        input = [char2index[c] for c in seq[:-1]]
        target = char2index[seq[-1]]
        input_batch.append(np.eye(vocab_size)[input])
        target_batch.append(target)

    return Variable(
        torch.Tensor(input_batch)), Variable(
        torch.LongTensor(target_batch))


# TextLSTM
class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden_size)
        self.W = nn.Parameter(torch.randn(
            [hidden_size, vocab_size]).type(dtype))
        self.b = nn.Parameter(torch.randn([vocab_size]).type(dtype))

    def forward(self, X):
        input = X.transpose(0, 1)

        # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        hidden_state = Variable(torch.zeros(1, len(X), hidden_size))
        # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1, len(X), hidden_size))

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        # model : [batch_size, n_class]
        model = torch.mm(outputs, self.W) + self.b
        return model


input_batch,target_batch = make_batch(seq_data)

model = TextLSTM()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()

    output = model(input_batch)
    loss = criterion(output,target_batch)
    if (epoch+1) % 100 == 0:
        print("Epoch:","%04d" %(epoch+1),'cost =','{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

predict = model(input_batch).data.max(1, keepdim=True)[1]
print([sen[:-1] for sen in seq_data], '->', [index2char[n.item()] for n in predict.squeeze()])