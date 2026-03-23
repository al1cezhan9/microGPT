# Architectural Patterns

## 1. Dataclass Config as Single Source of Truth

`GPTConfig` ([model.py:1-10](../../model.py#L1)) is a frozen dataclass that holds all hyperparameters (`block_size`, `vocab_size`, `n_layer`, `n_head`, `n_embd`, `dropout`). It is serialized into every checkpoint alongside the weights, so a checkpoint is fully self-describing. When loading a checkpoint in `app.py`, `train.py`, and `generate.py`, config is always reconstructed from the saved dict before instantiating the model — never hardcoded at the call site.

## 2. Hardware Detection — Consistent Three-Way Device Selection

Every file that touches PyTorch (`train.py`, `app.py`, `generate.py`, `streamlit.py`) uses the same three-way check:

```
cuda → mps (Apple Silicon) → cpu
```

The resolved `device` string is then passed into tensors, model `.to(device)`, and the autocast context. Mixed precision dtype also follows from device: `float16` for CUDA, `bfloat16` for MPS, `float32` for CPU.

## 3. Pre-LayerNorm Residual Blocks

Each transformer `Block` ([model.py](../../model.py)) normalizes inputs *before* the sublayer (pre-LN), then adds the residual:

```
x = x + attn(ln1(x))
x = x + ffwd(ln2(x))
```

This deviates from the original "Attention is All You Need" post-LN and matches the GPT-2 training-stability convention. A final `LayerNorm` is applied to the output of the last block before the language-model head.

## 4. Weight Tying

The token embedding matrix (`token_embedding_table`) and the final linear head (`lm_head`) share weights ([model.py](../../model.py)). This halves embedding parameter count and is standard practice for small language models.

## 5. Optional Attention-Weight Capture (Pass-Through Flag)

`Head.forward()` and `MultiHeadAttention.forward()` accept a `return_attn=False` flag. When `True`, raw attention weights are returned alongside the output tensor. `Block` propagates this flag through and returns `(x, attn_weights)`. The top-level `GPT.generate_stream()` uses this to surface per-token attention for the visualization endpoint, while `generate()` leaves it off for efficiency.

## 6. Streaming Generation Protocol (NDJSON over HTTP)

`GET /generate` ([app.py](../../app.py)) returns a `StreamingResponse` with `media_type="application/x-ndjson"`. Each line is a JSON object:

- Prompt tokens: `{"text": "<token>", "attn": []}` (empty attention)
- Generated tokens: `{"text": "<token>", "index": <int>, "attn": [[...], ...]}`

`attn` is a 2-D array of shape `(n_head, visible_seq_len)` — the attention weights from the *last* transformer block for the newly generated token. The frontend averages across heads and normalizes by the max weight to produce hover-highlight intensities.

## 7. Stateless BPE Encode/Decode

`encode()` and `decode()` in [bpe.py](../../bpe.py) are pure functions: they take vocabulary mappings as arguments and hold no internal state. Vocabulary (`vocab.json`) and merge rules (`merges.txt`) are loaded once at server startup and passed around as immutable data. This makes the tokenizer trivially thread-safe.

## 8. Dual Checkpoint Strategy

`train.py` maintains two checkpoint files in parallel:
- `checkpoint_latest.pth` — overwritten every evaluation; used for resuming.
- `checkpoint_best.pth` — overwritten only when validation loss improves; used for inference.

Both store model weights, optimizer state, scheduler state, current iteration, and the full metrics history, making training fully resumable with `Ctrl-C` safety (SIGINT handler saves before exit).

## 9. Greedy Autoregressive Context Window

Generation ([model.py](../../model.py), [generate.py](generate.py), [app.py](app.py)) crops the context to the last `block_size` tokens before each forward pass. There is no KV-cache — the full cropped sequence is re-encoded on every step. This is intentional for simplicity at the cost of O(n²) generation time.

## 10. Cached Model Loading in Streamlit

`streamlit.py` wraps model loading in `@st.cache_resource` so the model is loaded once per server process rather than on every user interaction. The pattern mirrors `app.py`'s module-level load, adapted to Streamlit's execution model.
