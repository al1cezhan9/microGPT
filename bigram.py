import torch
import torch.nn as nn
from torch.nn import functional as F

# --hyperparams--
batch_size = 64 # independent sequences processed in parallel
# previous 8 tokens predict 9th token
block_size = 256 # max context length
max_iters = 5000 # total num iterations of training 
eval_interval = 500 # how often we check loss during training
eval_iters = 200 
learning_rate = 3e-4
n_embd = 128 # embedding dimensions
n_layer = 4
n_head = 4
dropout = 0.2 # randomly sever connections during training -> ensemble
device = 'cuda' if torch.cuda.is_available() else 'cpu' # tragically untouched
print(f"Device: {device}")
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
  x,y = x.to(device), y.to(device)
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

# # normalizes rows
# class LayerNorma1d:
#   def __init__(self, dim, eps=1e-5):
#     self.eps = eps
#     self.gamma = torch.ones(dim)
#     self.beta = torch.zeros(dim)
#   def __call__(self, x):
#     xmean = x.mean(1, keepdim=True) # batch mean
#     xvar = x.var(1, keepdim=True) # batch var
#     xhat = (x-xmean)/torch.sqrt(xvar+self.eps) # normalize to unit var
#     self.out = self.gamma * xhat + self.beta
#     return self.out
#   def parameters(self):
#     return [self.gamma, self.beta]


class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.dropout = nn.Dropout(dropout)
    # tril is not a param of model, it's a buffer
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)
    wei = q @ k.transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    out = wei @ v
    return out
  
class MultiHeadAttention(nn.Module):
  def __init__(self, n_head, head_size):
    super().__init__()
    head_size = n_embd // n_head
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    return self.dropout(self.proj(out)) # project layer outcome back onto main highway
    
# done independently on a token-level basis
class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd), # x4dim from OG paper
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout),
    )
  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  def forward(self, x):
    # residual streams enable clean backprop
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(
      *[Block(n_embd, n_head=n_head) for _ in range(n_layer)],
    )
    self.ln_f = nn.LayerNorm(n_embd)
    # go from token embed -> logits
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B,T=idx.shape
    # idx and targets are both (B,T) tensor
    tok_embd = self.token_embedding_table(idx) # (B,T,n_embd)
    pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embd)
    x = tok_embd + pos_embd
    x = self.blocks(x)
    logits = self.lm_head(x) 

    if targets is None:
        loss = None
    else:
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
      # crop idx to last block_size token
      idx_cond = idx[:, -block_size:]
      # get predictions
      logits, loss = self(idx_cond)
      # take only the last time step prediction
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax: logits -> probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from distribution to get next token
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
model = BigramLanguageModel()
m = model.to(device)

# 1x1 tensor holding a 0, kickoff character
idx = torch.zeros((1, 1), dtype=torch.long)
print("="*20+"before training: "+"="*20)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

# time to train!
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

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
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print("="*20+"after training: "+"="*20)
output = decode(m.generate(idx, max_new_tokens=10000)[0].tolist())
print(output)
with open('generated_shakespeare.txt', 'w', encoding='utf-8') as f:
    f.write(output)

