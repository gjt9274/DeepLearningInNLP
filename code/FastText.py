import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os

from torchtext import data
from torchtext.vocab import Vectors
from torchtext.data import Iterator, BucketIterator, TabularDataset

# 用TorchText准备数据


def data_iter(data_path,vec_path, fix_length):
    TEXT = data.Field(sequential=True,
                      lower=True,
                      fix_length=fix_length,
                      batch_first=True)

    LABEL = data.Field(sequential=False, use_vocab=False)

    train, test = TabularDataset.splits(
        path=data_path,
        train='train.csv',
        test='test.csv',
        format='csv',
        fields=[('label', LABEL), ('title', None), ('text', TEXT)],
        skip_header=True
    )

    train_iter, test_iter = BucketIterator.splits(
        (train, test),  # 构建数据集所需的数据集
        batch_sizes=(8, 8),
        sort_within_batch=False,
        repeat=False
    )

    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)

    vectors = Vectors(name=vec_path, cache=cache)
    TEXT.build_vocab(train, vectors=vectors)
    vocab = TEXT.vocab

    return train_iter, test_iter, vocab

# train_iter,test_iter,vocab = data_iter('./data/ag_news_csv/',200)


class FastText(nn.Module):
    def __init__(self, vocab, vec_dim, label_size, hidden_size):
        super(FastText, self).__init__()
        # batch_size x len x vec_dim
        self.embed = nn.Embedding(len(vocab), vec_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True  # 权重不是固定的
        self.fc = nn.Sequential(
            nn.Linear(vec_dim, hidden_size),  # batch_size x len x hidden_size
            nn.BatchNorm1d(hidden_size), # 批规范化，batch_size x hidden_size
            nn.ReLU(inplace=True),  # batch_size x hidden_size
            nn.Linear(hidden_size, label_size) # batch_size x label_size
        )

    def forward(self, X):
        X = self.embed(X)  # batch_size x len x vec_dim
        out = self.fc(torch.mean(X, dim=1))

        return out


def train_model(model, train_iter, epoch, lr, batch_size):
    print("__________Train Begin__________")
    model.train()  # 将模型调成训练模式，会启用BatchNorm，dropout等

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoch):
        for i, batch in enumerate(train_iter):
            data, target = batch.text, batch.label - 1

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # print('Epoch:'+'%2d' % epoch +' batch_id:'
            #     +'%2d' %i +' Loss:' + '{:.6f}'.format(loss.item()/batch_size) )
    print('__________Train End__________')

def test_model(model,test_iter):
    print('__________Test Begin__________')
    model.eval() #将模型设置为测试模式

    correct = 0
    total = 0
    with torch.no_grad():
        for i ,batch in enumerate(test_iter):
            data,label = batch.text,batch.label-1
            output = model(data)
            predicted = torch.max(output.data,1)[1]

            total += label.size(0)
            correct += (predicted == label).sum().item()

            print('Accuracy of  the model on test set: %.2f %%' % (100*correct/total))
    print('__________Test End__________')


if __name__ == "__main__":
    data_path = './data/ag_news_csv/'
    vec_path = './vectors/glove.6B.300d.txt'
    fix_length = 50
    batch_size = 64
    epoch = 10
    embed_dim = 300
    lr = 0.001
    hidden_size = 200
    label_size = 4

    train_iter,test_iter ,vocab = data_iter(data_path,vec_path,fix_length)

    model = FastText(vocab,embed_dim,label_size,hidden_size)

    train_model(model,train_iter,epoch,lr,batch_size)

    test_model(model,test_iter)