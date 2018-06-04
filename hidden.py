from torch import nn

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.optim import Adam

from trainer import ModuleTrainer

def cuda(v):
  if torch.cuda.is_available():
    v = v.cuda()
  return v

MNIST_DATA_HOME="."

MNIST_DATA_HOME = 'D:\MLCode\datasets\MNIST'
mnist_train = datasets.MNIST(MNIST_DATA_HOME, train=True, transform=ToTensor(), download=True)
mnist_test = datasets.MNIST(MNIST_DATA_HOME, train=False, transform=ToTensor())



FMNIST_DATA_HOME = 'D:\MLCode\datasets\FashionMNIST'
FMNIST_DATA_HOME='.\FashionMNIST'
fmnist_train = datasets.FashionMNIST(FMNIST_DATA_HOME, train=True, transform=ToTensor(), download=True)
fmnist_test = datasets.FashionMNIST(FMNIST_DATA_HOME, train=False, transform=ToTensor(), download=True)

from VGG import VGG
net = VGG(1, [8, 8, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M'], 64, 10, batch_norm=False)
if torch.cuda.is_available():
  net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=0.0003, weight_decay=0)

model = ModuleTrainer(net)
model.compile(criterion, optimizer)
model.fit_dataset(fmnist_train, epochs=30, verbose=False)
model.evaluate_dataset(fmnist_test)



from LeNet import LeNet
from torch.optim.lr_scheduler import LambdaLR
net = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=0.001)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.3 * epoch))

model = ModuleTrainer(net)
model.compile(criterion, optimizer, scheduler)
model.fit_dataset(fmnist_train, epochs=10, num_examples=10000)
model.evaluate_dataset(fmnist_test)
