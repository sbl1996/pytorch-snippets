def load_word_to_vec(path):
  with open(path, 'r', encoding='utf-8') as f:
    word_to_vec = {}

    for line in f:
      line = line.strip().split()
      word = line[0]
      vec = np.array(line[1:], dtype=np.float)
      word_to_vec[word] = vec

  return word_to_vec

def get_embeddings(words, word_to_vec):
  embedding_dim = word_to_vec['the'].shape[0]
  n_words = len(words)
  word_to_ix = {}
  embeddings = np.zeros((n_words, embedding_dim), dtype=np.float)

  for i, word in enumerate(words):
    word_to_ix[word] = i
    embeddings[i] = word_to_vec[word]

  return word_to_ix, embeddings
