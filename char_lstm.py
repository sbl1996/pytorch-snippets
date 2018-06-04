from pathlib import Path
import string
import unicodedata
import time

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

def find_files(path, pattern):
  return Path(path).glob(pattern)

names_dir = './datasets/data/names'
pat = '*.txt'
print(list(find_files(names_dir, pat)))

letters = string.ascii_letters + " .,;'"
n_letters = len(letters)

def unicode_to_ascii(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
    and c in letters
  )
print(unicode_to_ascii('Ślusàrski'))

def read_lines(path):
  with open(path, encoding='utf-8') as f:
    return [ unicode_to_ascii(line) for line in f ]

categories = []
category_lines = {}

for f in find_files(names_dir, pat):
  category = f.name.split('.')[0]
  categories.append(category)
  lines = read_lines(f)
  category_lines[category] = lines

n_categories = len(categories)
print(category_lines['Italian'][:5])

def letter_to_tensor(letter):
  i = letters.index(letter)
  tensor = torch.zeros(1, n_letters)
  tensor[0][i] = 1
  return tensor

def line_to_tensor(line):
  letter_tensors = [letter_to_tensor(letter) for letter in line]
  return torch.cat(letter_tensors).view(len(line), 1, -1)

class RNN(nn.Module):

  def __init__(self, input_size, hidden_size, output_size):
    super(RNN, self).__init__()

    self.hidden_size = hidden_size

    self.gru = nn.GRU(input_size, hidden_size)
    self.h2o = nn.Linear(hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim=1)

  def init_hidden(self, batch_size):
    return torch.zeros(1, batch_size, self.hidden_size)

  def forward(self, input):
    batch_size = input.size()[1]
    hidden = self.init_hidden(batch_size)
    gru_out, h_n = self.gru(input, hidden)
    output = self.h2o(h_n).view(batch_size, -1)
    output = self.softmax(output)
    return output

def random_choice(l):
  return np.random.choice(l)

def random_training_example():
  i = np.random.randint(n_categories)
  category = categories[i]
  line = random_choice(category_lines[category])
  category_tensor = torch.tensor([i], dtype=torch.long)
  line_tensor = line_to_tensor(line)
  return category, line, category_tensor, line_tensor

def category_from_output(output):
  i = output.argmax().item()
  return categories[i], i

def time_since(since):
  now = time.time()
  s = now - since
  m = np.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

hidden_size = 128
rnn = RNN(n_letters, hidden_size, n_categories)

criterion = nn.NLLLoss()
lr = 0.005
optimizer = Adam(rnn.parameters(), lr)

n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

start = time.time()

for it in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_example()

    optimizer.zero_grad()

    output = rnn(line_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    
    current_loss += loss.item()

    # Print iter number, loss, name and guess
    if it % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '√' if guess == category else '× (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (it, it / n_iters * 100, time_since(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if it % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

plt.plot(all_losses)

confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = random_training_example()
    output = rnn(line_tensor)
    guess, guess_i = category_from_output(output)
    category_i = categories.index(category)
    confusion[category_i][guess_i] += 1

for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

