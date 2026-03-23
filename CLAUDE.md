# microGPT

A decoder-only GPT transformer trained on Shakespeare, built from scratch with PyTorch. Includes a custom BPE tokenizer, resumable training, and a FastAPI server with real-time attention visualization in the browser.

## Tech Stack

- **Python 3.10+**, **PyTorch** — transformer model and training
- **FastAPI + Uvicorn** — async HTTP server with streaming responses
- **Pydantic** — request validation
- **regex** — Unicode-aware BPE tokenizer
- **Vanilla JS / HTML5** — frontend (no framework)
- **Streamlit** — alternative demo UI
- **Docker** — containerization; deployed on Render.com

## Key Directories & Files

| Path | Purpose |
|------|---------|
| [model.py](model.py) | Transformer architecture (`GPTConfig`, `Head`, `MultiHeadAttention`, `Block`, `GPT`) |
| [train.py](train.py) | Training loop with checkpointing and mixed-precision |
| [bpe.py](bpe.py) | Byte-Pair Encoding tokenizer (learn merges + encode/decode) |
| [app.py](app.py) | FastAPI server; streaming `/generate` endpoint |
| [generate.py](generate.py) | CLI text generation utility |
| [static/index.html](static/index.html) | Browser UI with attention visualization |
| [streamlit.py](streamlit.py) | Alternative Streamlit demo interface |
| [plotting.py](plotting.py) | Plots training curves from checkpoint metrics |
| [testbpe.py](testbpe.py) | Roundtrip losslessness test for the tokenizer |
| `[2048V]model/` | Active model weights, vocab, and merges (`transformer.pth`, `vocab.json`, `merges.txt`) |
| `input.txt` | Shakespeare training corpus (~1.1 MB) |

## Commands

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Train** (auto-resumes from `checkpoint_latest.pth` if present)
```bash
python train.py
```

**Run the web server**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**CLI generation**
```bash
python generate.py
```

**Streamlit UI**
```bash
streamlit run streamlit.py
```

**Re-learn BPE merges** from `input.txt`
```bash
python bpe.py
```

**Plot training curves**
```bash
python plotting.py
```

**Test tokenizer**
```bash
python testbpe.py
```

**Docker**
```bash
docker build -t bardgpt .
docker run -p 10000:10000 bardgpt
```

## Additional Documentation

Check these files when working on the relevant subsystems:

| File | When to read |
|------|-------------|
| [.claude/docs/architectural_patterns.md](.claude/docs/architectural_patterns.md) | Architectural decisions, data-flow conventions, attention capture pattern, streaming protocol |
