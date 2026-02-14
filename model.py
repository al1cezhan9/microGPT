import torch
import torch.nn as nn
from torch.nn import functional as F

# --hyperparams--
block_size = 256 # context length, 1-256 predict 257
n_embd = 128 # embedding dimensions
n_layer = 4
n_head = 4
# randomly sever connections during training -> mimic ensemble
dropout = 0.2 

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.dropout = nn.Dropout(dropout)
    # tril is not a param of model, it's a buffer
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
  def forward(self, x, layer_past=None):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)
    # KV-caching!!! holy brilliance
    if layer_past is not None:
      # concat past KV with current
      past_k, past_v = layer_past
      k = torch.cat((past_k, k), dim=1)
      v = torch.cat((past_v, v), dim=1)
    # store this for the next token
    present = (k,v)
    wei = q @ k.transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    out = wei @ v
    return out, present
  
class MultiHeadAttention(nn.Module):
  def __init__(self, n_head, head_size):
    super().__init__()
    head_size = n_embd // n_head
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x, layer_past=None):
    head_outputs = []
    presents = []
    for i,h in enumerate(self.heads):
      past = layer_past[i] if layer_past is not None else None
      out, present = h(x, layer_past=past)
      head_outputs.append(out)
      presents.append(present)

    out = torch.cat(head_outputs, dim=-1)
    return self.dropout(self.proj(out)), presents # project layer outcome back onto main highway

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
  def forward(self, x, layer_past=None):
    attn_out, present = self.sa(self.ln1(x), layer_past=layer_past)
    # residual streams enable clean backprop
    x = x + attn_out
    x = x + self.ffwd(self.ln2(x))
    return x, present
  
class GPT(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.ModuleList(
      [Block(n_embd, n_head=n_head) for _ in range(n_layer)],
    )
    self.ln_f = nn.LayerNorm(n_embd)
    # go from token embed -> logits
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None, pasts=None):
    B,T=idx.shape
    tok_embd = self.token_embedding_table(idx)
    # adjust position since KV caching will warp this
    past_length = pasts[0][0].shape[1] if pasts is not None else 0
    pos_steps = torch.arange(past_length, past_length+T, device=idx.device)
    pos_embd = self.position_embedding_table(pos_steps)
    x = tok_embd + pos_embd
    new_pasts = []
    # KV-caching
    for i, block in enumerate(self.blocks):
      past = pasts[i] if pasts is not None else None
      x, present = block(x, layer_past=past)
      new_pasts.append(present)
    x = self.ln_f(x) # final layer norm
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
    return logits, loss, new_pasts
  
  def generate(self, idx, max_new_tokens, temperature=1.0):
    # list of caches of each layer
    pasts = [None] * self.n_layer
    # idx = (B, T) array of indices rn
    for _ in range(max_new_tokens):
      # only take very last tok as input since we have cache
      idx_cond = idx[:, -1:]
      # get predictions
      logits, loss, pasts = self(idx_cond)
      # take only the last time step prediction
      logits = logits[:, -1, :] # becomes (B, C)
      logits = logits / temperature # creativity!
      # apply softmax: logits -> probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from distribution to get next token
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
  

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
