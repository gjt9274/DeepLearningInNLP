import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

dtype = torch.FloatTensor
# S: Symbol that shows starting of decoding input
# E: Symbol that shows end of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is
# short than time steps

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
char2index = {c: i for i, c in enumerate(char_arr)}

seq_data = [
    ['man', 'women'], [
        'black', 'white'], [
        'king', 'queen'], [
        'girl', 'boy'], [
        'up', 'down'], [
        'high', 'low']]

# 超参数
n_step = 5
hidden_size = 128
vocab_size = len(char2index)
batch_size = len(seq_data)


def make_batch(seq_data):
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))

        input = [char2index[n] for n in seq[0]]
        output = [char2index[n] for n in ('S' + seq[1])]
        target = [char2index[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(vocab_size)[input])
        output_batch.append(np.eye(vocab_size)[output])
        target_batch.append(target)

    return Variable(
        torch.Tensor(input_batch)), Variable(
        torch.Tensor(output_batch)), Variable(
            torch.LongTensor(target_batch))

# seq2seq模型


class Seq2seq(nn.Module):
    def __init__(self):
        super(Seq2seq, self).__init__()
        self.enc_cell = nn.RNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            dropout=0.5)
        self.dec_cell = nn.RNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            dropout=0.5)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, enc_input, enc_hidden, dec_input):
        enc_input = enc_input.transpose(0, 1) # enc_input: [max_len(=n_step, time step), batch_size, n_class]
        dec_input = dec_input.transpose(0, 1) # dec_input: [max_len(=n_step, time step), batch_size, n_class]

        # enc_states : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        _, enc_states = self.enc_cell(enc_input, enc_hidden)
        # outputs : [max_len+1(=6), batch_size, num_directions(=1) * n_hidden(=128)]
        outputs, _ = self.dec_cell(dec_input, enc_states)

        model = self.fc(outputs) # model : [max_len+1(=6), batch_size, n_class]
        return model


input_batch, output_batch, target_batch = make_batch(seq_data)

model = Seq2seq()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5000):
    hidden = Variable(torch.zeros(1, batch_size, hidden_size))

    optimizer.zero_grad()
    output = model(input_batch, hidden, output_batch)
    output = output.transpose(0, 1)
    loss = 0
    for i in range(0, len(target_batch)):
        loss += criterion(output[i], target_batch[i])
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

# 测试


def translate(word):
    input_batch, output_batch, _ = make_batch([[word, 'P' * len(word)]])

    # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
    hidden = Variable(torch.zeros(1, 1, hidden_size))
    output = model(input_batch, hidden, output_batch)
    # output : [max_len+1(=6), batch_size(=1), n_class]

    predict = output.data.max(2, keepdim=True)[1]  # select n_class dimension
    decoded = [char_arr[i] for i in predict]
    end = decoded.index('E')
    translated = ''.join(decoded[:end])

    return translated.replace('P', '')


print('test')
print('man ->', translate('man'))
print('mans ->', translate('mans'))
print('king ->', translate('king'))
print('black ->', translate('black'))
print('upp ->', translate('upp'))
