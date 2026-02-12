import torch
import torch.nn as nn
from torch.nn import functional as F

# --hyperparams--
batch_size = 32 # independent sequences processed in parallel
# previous 8 tokens predict 9th token
block_size = 8 # max context length
max_iters = 5000 # total num iterations of training 
eval_interval = 300 # how often we check loss during training
eval_iters = 200
learning_rate = 1e-2
# no gpu RIP
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# read input shakespearan text
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# print(len(text))
# print(text[:100])

# get the vocab by sorting the set of the input text (str)
# set eliminates duplicates
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(vocab_size)
# print(''.join(chars))

# map char (str) to integer by index in the chars list
# ambigous ordering, losing a data dimension in proximity?
# tradeoff between size of vocab and dimension of embedding
ctoi = {ch:i for i,ch in enumerate(chars)}
itoc = {i:ch for i,ch in enumerate(chars)}
# lambda = mini-functions
encode = lambda s: [ctoi[c] for c in s]
decode = lambda l: ''.join([itoc[i] for i in l])
# enc = encode('hello! my name is alice')
# dec = decode(enc)
# print(enc)
# print(dec)

# encode entire dataset
# store it into a torch.Tensor (multi-d array optimized for computation)
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])

# split data into 90% train vs. 10% validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  # generate small batch of data (input & output)
  data = train_data if split == 'train' else val_data
  # get random position (-block_size) to account for valid starting positions
  # get {batch_size} number of these random offsets
  ix = torch.randint(len(data)-block_size, (batch_size,))
  # torch.stack stacks up into rows in 4x8 tensor
  # 32 examples, x and y just hold the end points
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,y

# xb, yb = get_batch('train')

@torch.no_grad() # tell pytorch we won't call backprop, efficiency 
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class BigramLanguageModel(nn.Module):
  
  def __init__(self):
    super().__init__()
    # embedding table W = matrix (vocab_size, vocab_size)
    # each token reads off logits (unnormalized scores) for next token via lookup table
    # row i of W contains logits for distr of next token given current token = i
    # this is the entire model bro
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    # idx and targets are both (B,T) tensor of ints
    logits = self.token_embedding_table(idx) # (B,T,C)

    if targets is None:
        loss = None
    else:
        # use built-in -log likelihood loss
        # python wants (B,C,T) instead
        # stretch out into 2D array
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    # idx = (B, T) array of indices rn
    for _ in range(max_new_tokens):
      # get predictions
      logits, loss = self(idx)
      # take only the last time step prediction
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax: logits -> probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from distribution to get next token
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
m = BigramLanguageModel()

# 1x1 tensor holding a 0, kickoff character
idx = torch.zeros((1, 1), dtype=torch.long)
print("="*20+"before training: "+"="*20)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

# time to train!
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# lol lowk just keep running this
for iter in range(max_iters):
  if iter % eval_interval == 0:
     losses = estimate_loss()
     print(f"step # {iter} || train loss: {losses['train']:.4f} || val loss: {losses['val']:.4f}")
  # sample batch of data
  xb, yb = get_batch('train')
  # eval the loss
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True) # zero out gradients
  loss.backward() # get grads for all params
  optimizer.step() # using grad to update params
# print(loss.item())

# check out improvements to predictions after training
idx = torch.zeros((1, 1), dtype=torch.long)
print("="*20+"after training: "+"="*20)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
# tokens still not talking to each other!



