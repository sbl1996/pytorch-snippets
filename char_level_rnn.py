import time
import math
from io import open
from pathlib import Path
import string
import unicodedata

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn

def find_files(path, pattern): return Path(path).glob(pattern)

print(list(find_files('datasets/data/names', '*.txt')))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicode_to_ascii(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
    and c in all_letters
  )
print(unicode_to_ascii('Ślusàrski'))

category_lines = {}
all_categories = []

def readlines(filename):
  with open(filename, encoding='utf-8') as f:
    return [unicode_to_ascii(line.strip()) for line in f]

for path in find_files('datasets/data/names', '*.txt'):
  category = path.name.split('.')[0]
  all_categories.append(category)
  lines = readlines(path)
  category_lines[category] = lines

n_categories = len(all_categories)

print(category_lines['Italian'][:5])

def letter_to_index(letter):
  return all_letters.find(letter)

def letter_to_tensor(letter):
  tensor = torch.zeros(1, n_letters)
  tensor[0][letter_to_index(letter)] = 1
  return tensor

def line_to_tensor(line):
  tensor = torch.zeros(len(line), 1, n_letters)
  for li, letter in enumerate(line):
    tensor[li][0][letter_to_index(letter)] = 1
  return tensor

print(letter_to_tensor('J'))

print(line_to_tensor('Jones').size())

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(RNN, self).__init__()

    self.hidden_size = hidden_size

    self.gr = nn.Linear(input_size + hidden_size, hidden_size)
    self.gu = nn.Linear(input_size + hidden_size, hidden_size)
    self.li = nn.Linear(input_size, hidden_size)
    self.lh = nn.Linear(hidden_size, hidden_size)
    self.lo = nn.Linear(hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, input, hidden):
    combined = torch.cat((input, hidden), 1)
    r = torch.sigmoid(self.gr(combined))
    u = torch.sigmoid(self.gu(combined))
    n = torch.tanh(self.li(input) + r * self.lh(hidden))
    hidden = (1 - u) * n + u * hidden
    output = self.lo(hidden)
    output = self.softmax(output)
    return output, hidden

  def initHidden(self):
    return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

input = line_to_tensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)

def category_from_output(output):
  top_i = output.argmax()
  category_i = top_i.item()
  return all_categories[category_i], category_i

def random_choice(l):
  return np.random.choice(l)

def random_training_example():
  category = random_choice(all_categories)
  line = random_choice(category_lines[category])
  category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
  line_tensor = line_to_tensor(line)
  return category, line, category_tensor, line_tensor

criterion = nn.NLLLoss()

learning_rate = 0.005

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for it in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

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

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = random_training_example()
    output = evaluate(line_tensor)
    guess, guess_i = category_from_output(output)
    category_i = all_categories.index(category)
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

