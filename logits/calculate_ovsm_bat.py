import numpy as np
import os
import torch
from model import GPTConfig, GPT
from scipy.optimize import linprog
from tqdm import tqdm
import itertools
import math

import multiprocessing
from functools import partial

np.random.seed(15)
samples = np.random.randint(0, 100, 100)
byte_offsets = np.random.randint(0, 500, 100)
pairs = np.column_stack((samples, byte_offsets))

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

LOCAL_MODEL_DIR = '../new-16-layer-models'

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def load_local_model(n_bottleneck):
    ckpt_path = os.path.join(LOCAL_MODEL_DIR, f'ckpt-{n_bottleneck}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'")

    print(f"Loading local model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print("Local model loaded successfully.")
    return model

def find_true_support_sequentially(p_hat, W, threshold):
    delta = np.log(1 / (1 - threshold))
    v, d = W.shape

    A_eq_sum = np.ones((1, v))
    b_eq_sum = np.array([1])
    A_eq_W = W.T
    b_eq_W = W.T @ p_hat
    A_eq = np.vstack([A_eq_sum, A_eq_W])
    b_eq = np.concatenate([b_eq_sum, b_eq_W])
    upper_bounds = p_hat * np.exp(delta)
    bounds = list(zip(np.zeros(v), upper_bounds))

    support = []
    for k in range(v):
        if (p_hat[k] >= threshold):
            support.append(k)
        else:
            c = np.zeros(v)
            c[k] = 1.0
            tol = 1e-10
            result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={'disp': False, "dual_feasibility_tolerance": tol, "primal_feasibility_tolerance": tol,"objective_tolerance": tol, "presolve": True })
            if result.success:
                sum_ok    = abs(np.sum(result.x) - 1.0) <= tol
                zero_ok   = abs(result.x[k]) <= tol
                Wok       = np.allclose(W.T @ result.x, W.T @ p_hat, atol=tol, rtol=0)
                ub_ok     = np.all((result.x - p_hat * np.exp(delta)) <= tol)
                out_of_support = sum_ok and zero_ok and Wok and ub_ok
                if not out_of_support:
                    support.append(k)
            else:
                support.append(k)
    return support

def process_pair(pair, bns, W_matrices, logits, threshold):
    sample, byte_offset_in_gt = pair
    pair_totals = np.zeros(len(bns))
    
    for i, bn in enumerate(bns):
        p_hat = softmax(logits[i][sample, byte_offset_in_gt, :])
        support = find_true_support_sequentially(p_hat=p_hat, W=W_matrices[i], threshold=threshold)
        if (len(support) < 256):
            pair_totals[i] = 1 - np.sum(p_hat[support])
        
    return pair_totals

if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)

    bns = [4, 8, 12, 16, 32, 128, 512]
    W_matrices = [load_local_model(n_bottleneck=bn).lm_head.weight.detach().cpu().numpy() for bn in bns]
    logits = [np.load(f'gpt_logits_{bn}_bos.npy') for bn in bns]

    threshold = 0.2

    worker_func = partial(
        process_pair, 
        bns=bns, 
        W_matrices=W_matrices, 
        logits=logits, 
        threshold=threshold
    )

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(worker_func, pairs), total=len(pairs), desc="Checking Samples"))

    totals = np.sum(results, axis=0)
    averages = totals / len(pairs)
    
    print("-" * 30)
    for i in range(len(bns)):
        print(f"{bns[i]}: {averages[i]}")