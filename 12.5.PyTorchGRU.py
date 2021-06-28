import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import data, datasets
import random

random.seed(5)
torch.manual_seed(5)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 10
Batch_size = 64

# data load
text = data.Field(sequential = True, batch_first = True, lower = True) # batch_first : 입력 텐서의 첫번째 차원값이 batch_size
label = data.Field(sequential = False, batch_first = True)

trainset, testset = datasets.IMDB.splits(text, label)
print(trainset.fields)
print(vars(trainset[0]))

text.build_vocab(trainset, min_freq = 5)
label.build_vocab(trainset)

vocab_size = len(text.vocab)
n_classes = 2
print(vocab_size)
# print(text.vocab.stoi)  # 단어와 각 단어의 정수 인덱스가 저장된 딕셔너리 객체 접근 가능

trainset, valset = trainset.split(split_ratio = 0.8)

train_iter, val_iter, test_iter = data.BucketIterator.splits((trainset, valset, testset),
                                                                batch_size = Batch_size,
                                                                shuffle = True,
                                                                repeat = False)

# batch = next(iter(train_iter))
# print(batch.text.shape)    #torch.Size([64, 457])
# batch = next(iter(train_iter))
# print(batch.text.shape)    #torch.Size([64, 903])

# class def
class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(GRU, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0)) # 첫번째 히든 스테이트를 0벡터로 초기화
        x, _ = self.gru(x, h_0)  # GRU의 리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        h_t = x[:,-1,:] # (배치 크기, 은닉 상태의 크기)의 텐서로 크기가 변경. 즉, 마지막 time-step의 은닉 상태만 가져옴.
        self.dropout(h_t)
        logit = self.out(h_t)  # (배치 크기, 은닉 상태의 크기) -> (배치 크기, 출력층의 크기)
        return logit

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

# model def    
model = GRU(1, 256, vocab_size, 128, n_classes, 0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# train def
def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)  
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

# evaluation def
def evaluate(model, val_iter):
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1) 
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

# train
best_val_loss = None
for e in range(1, training_epochs+1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

    # model save
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss

# predict
model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter)