import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import string
import json
import argparse
import itertools
import math
from scipy.optimize import linprog
from unargmaxable import candidate_is_bounded

bottlenecks = [4, 8, 12, 16, 32, 128, 512]

def get_weights(n_bottleneck):
    LOCAL_MODEL_DIR = '../new-16-layer-models'
    from model import GPTConfig, GPT
    import torch
    import os
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )  
    ckpt_path = os.path.join(LOCAL_MODEL_DIR, f'ckpt-{n_bottleneck}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    return model.lm_head.weight.detach().cpu().numpy()

def get_weights_normalize(n_bottleneck):
    LOCAL_MODEL_DIR = '../new-16-layer-models'
    from model import GPTConfig, GPT
    import torch
    import os
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )  
    ckpt_path = os.path.join(LOCAL_MODEL_DIR, f'ckpt-{n_bottleneck}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    weight_matrix = model.lm_head.weight.detach().cpu().numpy()
    assert weight_matrix.shape == (256, n_bottleneck)
    norms = np.linalg.norm(weight_matrix, axis=1, keepdims=True)
    return weight_matrix / norms

ascii_control_labels = [
    'NUL', 'SOH', 'STX', 'ETX', 'EOT', 'ENQ', 'ACK', 'BEL',
    'BS', 'HT', 'LF', 'VT', 'FF', 'CR', 'SO', 'SI',
    'DLE', 'DC1', 'DC2', 'DC3', 'DC4', 'NAK', 'SYN', 'ETB',
    'CAN', 'EM', 'SUB', 'ESC', 'FS', 'GS', 'RS', 'US'
]

def escape_for_latex(label: str) -> str:
    latex_escapes = {
        '\\': r'\textbackslash ',
        '#': r'\#',
        '$': r'\$',
        '%': r'\%',
        '^': r'\^{}',
        '&': r'\&',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '<': r'\textless ',
        '>': r'\textgreater ',
    }
    return ''.join(latex_escapes.get(c, c) for c in label)

def generate_labels_with_tokenizer(tokenizer):
    labels = []
    printable_chars = string.digits + string.ascii_letters + string.punctuation + ' '
    for byte_value in range(256):
        token_id = byte_value + 64
        raw_label = tokenizer.decode([token_id])
        if raw_label == '\n':
            labels.append('\\n')
        elif raw_label == '\r':
            labels.append('\\r')
        elif raw_label == '\t':
            labels.append('\\t')
        elif raw_label in printable_chars:
            labels.append(raw_label)
        else:
            labels.append(f'\\x{byte_value:02x}')
    return labels

def render_label(byte: int, label: str) -> str:
    if byte < 0x20:
        return rf'\fbox{{{ascii_control_labels[byte]}}}'
    elif byte == 0x20:
        return r'\fbox{SP}'
    elif byte == 0x7F:
        return r'\fbox{DEL}'
    elif 0x21 <= byte <= 0x7E:
        return escape_for_latex(label) + r'\ '
    else:
        return rf'\fbox{{{byte:02X}}}'

tokenizer = AutoTokenizer.from_pretrained("../EvaByte_local", local_files_only=True, trust_remote_code=True)
byte_labels = generate_labels_with_tokenizer(tokenizer)

if __name__ == '__main__':
    weights = [get_weights_ce(n_bottleneck) for n_bottleneck in bottlenecks]

    for i in range(len(bottlenecks)):
        total = 0
        possible = []
        impossible = []

        for k in range(256):
            if not candidate_is_bounded(k, weights[i])['is_bounded']:
                total += 1
                possible.append(k)
            else:
                impossible.append(k)

        print(bottlenecks[i])
        print(256 - total)

        latex_safe = ''.join(
            render_label(k, byte_labels[k]) + r'\allowbreak{}'
            for k in impossible
        )

        print(f"Unargmaxable: {latex_safe}")
        print("-" * 30)