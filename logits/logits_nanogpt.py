import os
import json
import argparse
import torch
import numpy as np
from contextlib import nullcontext
from model import GPTConfig, GPT

LOCAL_MODEL_DIR = '16-layer-models' 

PROMPT_FILE = 'prompts.jsonl'
GROUND_TRUTH_FILE = 'ground_truths.jsonl'
COMPLETION_LENGTH = 500 
OUTPUT_FILE_TEMPLATE = 'gpt_logits_{n_bottleneck}.npy'

SEED = 1337
DEVICE = 'cuda'

def setup_parser():
    parser = argparse.ArgumentParser(description="Extract logits from a local bottlenecked GPT model.")
    parser.add_argument('--n_bottleneck', type=int, required=True)
    return parser

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

def main():
    parser = setup_parser()
    args = parser.parse_args()
    n_bottleneck = args.n_bottleneck

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32 
    
    local_model = load_local_model(n_bottleneck)
    vocab_size = local_model.config.vocab_size
    
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"Prompt file not found at '{PROMPT_FILE}'")
    if not os.path.exists(GROUND_TRUTH_FILE):
        raise FileNotFoundError(f"Ground truth file not found at '{GROUND_TRUTH_FILE}'")
        
    with open(PROMPT_FILE, 'r', encoding='utf-8') as p_file:
        prompts = p_file.readlines()
        num_prompts = len(prompts)
        
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as gt_file:
        ground_truths = {json.loads(line)['index']: json.loads(line)['ground_truth'] for line in gt_file}

    output_file = OUTPUT_FILE_TEMPLATE.format(n_bottleneck=n_bottleneck)
    print(f"Initializing NumPy array with shape: ({num_prompts}, {COMPLETION_LENGTH}, {vocab_size})")
    all_logits = np.zeros((num_prompts, COMPLETION_LENGTH, vocab_size), dtype=np.float32)

    ctx = nullcontext() if DEVICE == 'cpu' else torch.amp.autocast(device_type=DEVICE, dtype=compute_dtype)

    for i, line in enumerate(prompts):
        record = json.loads(line)
        prompt_index = record['index']
        prompt_text = record['prompt']
        
        if prompt_index not in ground_truths:
            print(f"Warning: No ground truth found for prompt index {prompt_index}. Skipping.")
            continue
            
        ground_truth_text = ground_truths[prompt_index]
        ground_truth_tokens = list(ground_truth_text.encode('utf-8'))
        eval_length = min(COMPLETION_LENGTH, len(ground_truth_tokens))

        if eval_length == 0:
            print(f"Warning: Ground truth for prompt index {prompt_index} is empty. Skipping.")
            continue

        print(f"\n--- Processing Prompt Index: {prompt_index} ({i+1}/{num_prompts}) ---", flush=True)
        
        local_input_ids = list(prompt_text.encode('utf-8'))
        
        with torch.no_grad():
            with ctx:
                for step in range(eval_length):
                    local_input_tensor = torch.tensor([local_input_ids], dtype=torch.long, device=DEVICE)
                    logits_tensor, _ = local_model(local_input_tensor)
                    logits_tensor = logits_tensor[:, -1, :].squeeze()
                    
                    all_logits[i, step, :] = logits_tensor.cpu().numpy()

                    next_token_id = ground_truth_tokens[step]
                    
                    local_input_ids.append(next_token_id)

    print(f"\nSaving logits to '{output_file}'...")
    np.save(output_file, all_logits)
    print("Logit extraction complete.")

if __name__ == '__main__':
    main()
