# GPT From Scratch

A character-level GPT built from scratch in PyTorch: multi-head self-attention, transformer blocks, and text generation.

Built by following Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture.

## What's inside

Everything lives in a single file, `gpt.py`:

- **Tokenizer**: character-level encode/decode
- **Data loader**: random batches of context/target pairs (90/10 train/val split)
- **Self-attention head**: key, query, value projections with causal masking
- **Multi-head attention**: several heads in parallel plus an output projection
- **Feed-forward network**: expands to 4x width, then projects back
- **Transformer block**: attention and feed-forward with residual connections and LayerNorm
- **GPT model**: token + positional embeddings, stacked blocks, final LayerNorm, language-model head
- **Training loop**: AdamW with periodic train/val loss evaluation
- **Generation**: autoregressive sampling with context cropping

## Setup

```bash
git clone https://github.com/arsal1947/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Put your training text in a file named `input.txt` in the project folder (for example, the [tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset).

## Run

```bash
python gpt.py
```

This trains the model, prints train/val loss every 500 steps, and then generates 500 characters of text.

## Hyperparameters

Set at the top of `gpt.py`:

| Name | Value |
|---|---|
| batch_size | 32 |
| block_size (context length) | 64 |
| n_embd | 128 |
| n_head | 4 |
| n_layer | 4 |
| dropout | 0.1 |
| learning_rate | 1e-3 |
| max_iters | 3000 |

These are sized to run on a laptop CPU. With a GPU you can scale up (for example `block_size=256`, `n_embd=384`, `n_head=6`, `n_layer=6`, `learning_rate=3e-4`).

## Results

- Parameters: ~0.8M
- Final train loss: `<fill in>`
- Final val loss: `<fill in>`

Sample output:

```
<paste a snippet of generated text here>
```

## Credits

Based on Andrej Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture and the [nanoGPT](https://github.com/karpathy/nanoGPT) project.
