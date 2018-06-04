import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torch.utils.data import DataLoader
import torchvision.utils as vutils
%matplotlib inline

MNIST_HOME = '../datasets/MNIST'

MNIST_HOME = '.'
transform = Compose([
  Resize(32),
  ToTensor(),
  Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
mnist = datasets.MNIST(MNIST_HOME, transform=transform, download=True)

cifar = datasets.CIFAR10(".", transform=transform, download=True)

BATCH_SIZE = 64
N_HIDDEN = 10

def cuda(x):
  if torch.cuda.is_available():
    return x.cuda()
  return x

def make_hidden(batch_size):
  z = torch.randn(batch_size, N_HIDDEN, 1, 1)
  return z

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

G = nn.Sequential(                  
    nn.ConvTranspose2d(N_HIDDEN, 128, 4, 1, 0, bias=False),    
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(True),
    nn.ConvTranspose2d(16, 3, 3, 1, 1, bias=False),
    nn.Tanh(),
)
G.apply(weights_init)
G = cuda(G)

D = nn.Sequential(                      
    nn.Conv2d(3, 16, 3, 1, 1, bias=False),
    nn.LeakyReLU(inplace=True),
    nn.Conv2d(16, 32, 4, 2, 1, bias=False),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(inplace=True),
    nn.Conv2d(32, 64, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(inplace=True),
    nn.Conv2d(64, 128, 4, 2, 1, bias=False),
    nn.BatchNorm2d(128),
    nn.Conv2d(128, 1, 4, 1, 0, bias=False),
    nn.Sigmoid(),
)
D.apply(weights_init)
D = cuda(D)

opt_D = torch.optim.Adam(D.parameters(), lr=0.0001)
opt_G = torch.optim.Adam(G.parameters(), lr=0.0001)

criterion = cuda(nn.BCELoss())

epoches = 40
for epoch in range(epoches):
  dataloader = DataLoader(cifar, batch_size=BATCH_SIZE, shuffle=True)
  for i, data in enumerate(dataloader):
    D.zero_grad()
    x, _ = data
    batch_size = len(x)
    x = cuda(Variable(x))

    real_y = D(x)
    D_x = real_y.data.mean()
    label = torch.FloatTensor(batch_size, 1, 1, 1).fill_(1)
    labelv = cuda(Variable(label))
    errD_real = criterion(real_y, labelv)

    z = make_hidden(batch_size)
    z = cuda(Variable(z))
    fake_x = G(z)

    fake_y = D(fake_x.detach())
    label = torch.FloatTensor(batch_size, 1, 1, 1).fill_(0)
    labelv = cuda(Variable(label))
    errD_fake = criterion(fake_y, labelv)    
    
    errD = errD_real + errD_fake
    errD.backward()
    D_G_z1 = fake_y.data.mean()

    opt_D.step()
    

    G.zero_grad()
    label = torch.FloatTensor(batch_size, 1, 1, 1).fill_(1)
    labelv = cuda(Variable(label))
    fake_y = D(fake_x)
    errG = criterion(fake_y, labelv)
    errG.backward()
    D_G_z2 = fake_y.data.mean()
    opt_G.step()

    if i % 100 == 0:
      print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
        % (epoch, epoches, i, len(dataloader),
           errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

    if i % 100 == 0:
      fake_x = G(z)
      show(vutils.make_grid(fake_x.data.cpu(), normalize=True, nrow=8))