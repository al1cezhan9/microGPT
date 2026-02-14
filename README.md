# Micro-GPT Shakespeare

A character-level Transformer built from scratch to distill the stylistic essence of the Bard. This project implements a decoder-only architecture (GPT-style) that you can train, save, and serve via a web API.


## File Structure

* **`model.py`**: The core architecture. Contains the `GPT` class, `Block` layers, and `MultiHeadAttention`.
* **`train.py`**: The training engine. Handles data loading, loss estimation, and **persistent checkpointing**.
* **`generate.py`**: A simple script to load the `.pth` file and print text to your terminal.
* **`app.py`**: A FastAPI backend that serves the model for web interfaces.
* **`transformer.pth`**: Saved weights (includes both model and optimizer states).
* **`vocab.json`**: The character-to-integer mapping key.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install torch fastapi uvicorn pydantic
```
### 1. Training
Run this to start training. If a transformer.pth exists, it will automatically resume from where you left off.

```bash
python train.py
```

### 1. Training
Run this to see the web app in action. 

```bash
streamlit run app.py
```

