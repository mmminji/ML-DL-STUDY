from math import hypot
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

# data load
mnist_train = dsets.MNIST(root = './',                              # 다운로드 경로
                            train = True,                           # 훈련 데이터 지정
                            transform = transforms.ToTensor(),      # 텐서로 변환
                            download = True)
mnist_test = dsets.MNIST(root = './',                    
                            train = False,                          
                            transform = transforms.ToTensor(),    
                            download = True)
data_loader = torch.utils.data.DataLoader(dataset = mnist_train,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            drop_last = True)
total_batch = len(data_loader)

# class def
class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1),
                                            torch.nn.ReLU(),
                                            torch.nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
                                            torch.nn.ReLU(),
                                            torch.nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.fc = torch.nn.Linear(7*7*64, 10, bias = True)

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# model def        
model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)                     # 비용함수
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)   # 옵티마이저

# train
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch : {:>4} cost = {:.9}'.format(epoch+1, avg_cost))

# predict
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    pred = model(X_test)
    correct_prediction = torch.argmax(pred, 1) == Y_test
    acc = correct_prediction.float().mean()
    print('Accuracy : ', acc.item())