import torch
import torch.nn as nn
from torch.nn import functional as F

# --hyperparams--
block_size = 256 
n_embd = 128 
n_layer = 4
n_head = 4
dropout = 0.2 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Linear layers map from total embedding dimension to specific head dimension
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Buffer for the causal mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, layer_past=None):
        B, T, C = x.shape # T is the current input length (1 during generation)
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=1) # (B, T_total, head_size)
            v = torch.cat((past_v, v), dim=1) # (B, T_total, head_size)
        
        present = (k, v)
        T_total = k.shape[1]

        # Compute attention scores ("affinities")
        # Scale by the square root of head_size (dk)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        
        # Apply causal mask: we only mask the current query against future keys
        # When generating, T=1, so we only need the last row of the mask
        wei = wei.masked_fill(self.tril[T_total-T:T_total, :T_total] == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, head_size)
        return out, present

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        # Total head_size * n_head must equal n_embd
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer_past=None):
        head_outputs = []
        presents = []
        for i, h in enumerate(self.heads):
            # Extract specific head cache if it exists
            past = layer_past[i] if layer_past is not None else None
            out, present = h(x, layer_past=past)
            head_outputs.append(out)
            presents.append(present)

        out = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, presents

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
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
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, present

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, pasts=None):
        B, T = idx.shape
        
        # Determine current offset for positional embeddings based on cache
        past_length = 0
        if pasts is not None and pasts[0] is not None:
            # pasts[layer][head][0] is the Key tensor
            past_length = pasts[0][0][0].shape[1] 

        tok_embd = self.token_embedding_table(idx) # (B, T, n_embd)
        
        # Handle positional embeddings with offset
        pos_steps = torch.arange(past_length, past_length + T, device=idx.device) % block_size
        pos_embd = self.position_embedding_table(pos_steps) # (T, n_embd)
        
        x = tok_embd + pos_embd
        
        new_pasts = []
        for i, block in enumerate(self.blocks):
            past = pasts[i] if pasts is not None else None
            x, present = block(x, layer_past=past)
            new_pasts.append(present)
            
        x = self.ln_f(x)
        logits = self.lm_head(x) 

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss, new_pasts

    def generate(self, idx, max_new_tokens, temperature=1.0):
      # Ensure starting idx isn't longer than block_size
      idx = idx[:, -block_size:] 
      logits, _, pasts = self(idx)
      
      for _ in range(max_new_tokens):
          logits_last = logits[:, -1, :] / temperature
          probs = F.softmax(logits_last, dim=-1)
          idx_next = torch.multinomial(probs, num_samples=1)
          
          idx = torch.cat((idx, idx_next), dim=1)
          
          # Check if we've hit the context limit
          # past_length is stored in the first K tensor
          current_pos = pasts[0][0][0].shape[1]
          
          if current_pos >= block_size:
              # If we hit the limit, we can't easily use the KV cache 
              # with absolute positional embeddings without complex logic.
              # Simple fix: Clear cache and crop idx (expensive but safe)
              logits, _, pasts = self(idx[:, -block_size:])
          else:
              logits, _, pasts = self(idx_next, pasts=pasts)
              
      return idx