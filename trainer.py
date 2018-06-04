import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import math

class ModuleTrainer(object):
  def __init__(self, net):
    super(ModuleTrainer, self).__init__()
    self.net = net

  def compile(self, loss, optimizer, lr_scheduler=None):
    self.criterion = loss
    self.optimizer = optimizer
    self.lr_scheduler = lr_scheduler

  def fit(self, ite, epochs=20, verbose=True):
    num_batches = len(data_loader)
    batch_size = data_loader.batch_size
    m = len(data_loader.dataset)
    for epoch in range(epochs):
      if self.lr_scheduler:
        self.lr_scheduler.step()
      losses = []
      i = 0
      loss_avg = 0
      for batch in data_loader:
        i += 1
        batch_X, batch_Y = batch
        self.net.zero_grad()
        batch_X = cuda(Variable(batch_X))
        output = self.net(batch_X)
        batch_Y = cuda(Variable(batch_Y))
        loss = self.criterion(output, batch_Y)
        losses.append(float(loss))
        loss.backward()
        self.optimizer.step()
        loss_avg = (loss_avg * (i - 1) + losses[-1]) / i
        if verbose:
          print('\rEpoch %3d/%3d %5d/%5d\tloss: %f' % (epoch + 1, epochs, min(i*batch_size, m), m, loss_avg), end='')
      print()

  def fit_dataset(self, dataset, batch_size=32, epochs=20, verbose=True, num_examples=-1):
    if num_examples == -1:
      data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
      m = len(dataset)
    else:
      sampler = SubsetRandomSampler(range(num_examples))
      data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
      m = num_examples
    num_batches = math.ceil(m / batch_size)
    loss_avgs = []
    for epoch in range(epochs):
      if self.lr_scheduler:
        self.lr_scheduler.step()
      losses = []
      i = 0
      loss_avg = 0
      for batch in data_loader:
        i += 1
        batch_X, batch_Y = batch
        self.net.zero_grad()
        batch_X = cuda(Variable(batch_X))
        output = self.net(batch_X)
        batch_Y = cuda(Variable(batch_Y))
        loss = self.criterion(output, batch_Y)
        losses.append(float(loss))
        loss.backward()
        self.optimizer.step()
        loss_avg = (loss_avg * (i - 1) + losses[-1]) / i
        if verbose:
          print('\rEpoch %3d/%3d %5d/%5d\tloss: %f' % (epoch + 1, epochs, min(i*batch_size, m), m, loss_avg), end='')
      loss_avgs.append(loss_avg)
      if verbose:
        print()
      else:
        print('Epoch %3d/%3d \tloss: %f' % (epoch + 1, epochs, loss_avgs[-1]))
    return loss_avgs


  def evaluate_dataset(self, dataset, batch_size=32):
    self.net.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_correct = 0
    for batch in data_loader:
      batch_X, batch_Y = batch
      batch_X = cuda(Variable(batch_X))
      batch_Y = cuda(batch_Y)
      out = self.net(batch_X)
      Y = torch.max(out.data, dim=1)[1]
      n_correct += (Y == batch_Y).sum()
    self.net.train()
    return n_correct / len(dataset)