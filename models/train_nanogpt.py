import os
import time
import math
import pickle
from contextlib import nullcontext
import argparse

import numpy as np
import torch
from model import GPTConfig, GPT
from transformers import GPT2Tokenizer
from datasets import load_dataset
import wandb
# -----------------------------------------------------------------------------
# Model and Training Hyperparameters and Configuration
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_bottleneck", 
    required=True, 
    type=int,
)

args = parser.parse_args()

eval_interval = 1000
eval_iters = 200
log_interval = 100 # don't print too too often
gradient_accumulation_steps = 8
batch_size = 24
block_size = 1024  # context of up to 1024 previous characters

# Baby GPT model configuration
n_layer = 16
n_head = 16
n_embd = 1024
n_bottleneck = args.n_bottleneck
dropout = 0.1
vocab_size = 256  # 256 possible byte values
bias = False

# AdamW optimizer settings
learning_rate = 3e-4
max_iters = 100000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 1e-4 
beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1
warmup_iters = 1000

config = {
    "n_bottleneck":            n_bottleneck,
    "eval_interval":           eval_interval,
    "eval_iters":              eval_iters,
    "log_interval":            log_interval,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "batch_size":              batch_size,
    "block_size":              block_size,
    "n_layer":                 n_layer,
    "n_head":                  n_head,
    "n_embd":                  n_embd,
    "dropout":                 dropout,
    "vocab_size":              vocab_size,
    "bias":                    bias,
    "learning_rate":           learning_rate,
    "max_iters":               max_iters,
    "lr_decay_iters":          lr_decay_iters,
    "min_lr":                  min_lr,
    "beta1":                   beta1,
    "beta2":                   beta2,
    "weight_decay":            weight_decay,
    "warmup_iters":            warmup_iters,
}

model_dir = 'out-fineweb-char/24-layer-models'
checkpoint_location = os.path.join(model_dir, 'ckpt-' + str(n_bottleneck) + '.pt')

data_dir = 'out-fineweb-char'

print(f'Checkpoint location: {checkpoint_location}')

wandb.init(project='fineweb-char-' + str(n_bottleneck), config=config)

# To avoid overfitting, only save when val improves
always_save_checkpoint = False

# System settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
os.makedirs(data_dir, exist_ok=True)
torch.manual_seed(1337)

# Set up device context
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data Loading and Preprocessing with Caching
# -----------------------------------------------------------------------------
# Define a cache path for the pre-encoded data
data_cache_path = os.path.join(data_dir, 'data_cache.bin.npy')

# Check if pre-encoded data exists
if os.path.exists(data_cache_path):
    print(f"Loading pre-encoded data from cache: {data_cache_path}")
    data = np.load(data_cache_path, mmap_mode='r').astype(np.uint8)
else:
    print("Pre-encoded data cache not found. Processing from source...")
    print("Loading raw dataset from parquet...")
    # Load the tokenizer locally
    tokenizer = GPT2Tokenizer.from_pretrained("", local_files_only=True)

    # Load the dataset from the specified parquet file
    ds = load_dataset("parquet", data_files="000_0000.parquet", split="train")

    # Concatenate all text documents into one large string and then encode
    print("Encoding dataset (this will take a while but only happens once)...")
    encoded_data = b''.join([doc['text'].encode('utf-8') for doc in ds])
    data = np.frombuffer(encoded_data, dtype=np.uint8)

    # Save the processed data to the cache file for next time
    print(f"Saving pre-encoded data to cache: {data_cache_path}")
    np.save(data_cache_path, data)

print(f"Total data size: {len(data) / 1e6:.2f}M bytes")

# Split into training and validation sets (90% train, 10% val)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Convert to torch tensors
train_data = torch.from_numpy(train_data.astype(np.int64))
val_data = torch.from_numpy(val_data.astype(np.int64))
print("Data preparation complete.")

# Data loader function
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Model Initialization and Optimizer
# -----------------------------------------------------------------------------
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=vocab_size, dropout=dropout, n_bottleneck=n_bottleneck)

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)
print(f"Initialized a new model with {model.get_num_params()/1e6:.2f}M parameters")

print(model)

# Compile the model for a significant speedup
if compile:
    print("Compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

# Create AdamW optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

# -----------------------------------------------------------------------------
# Helper Functions for Training
# -----------------------------------------------------------------------------
# Learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
X, Y = get_batch('train') # fetch the first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
iter_num = 0
best_val_loss = 1e9

print("\nStarting training loop...")
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if lr_decay_iters > 0 else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        wandb.log({
            'eval/train_loss': losses['train'],
            'eval/val_loss': losses['val'],
            'eval/best_val_loss': best_val_loss,
        }, step=iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': gptconf,
                }
                print(f"saving checkpoint to {model_dir}")
                torch.save(checkpoint, checkpoint_location)

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        loss.backward()

    # clip the gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # step the optimizer and scaler if training in fp16
    optimizer.step()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:e}")

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

artifact = wandb.Artifact('fineweb-char-' + str(n_embd) + '-model', type="model")
artifact.add_file(checkpoint_location)

# log it to W&B
wandb.log_artifact(artifact)
wandb.finish()
