import os
import json
import argparse
import torch
import numpy as np
from contextlib import nullcontext
from transformers import AutoTokenizer, AutoModelForCausalLM

LOCAL_EVABYTE_PATH = "EvaByte_local" 

PROMPT_FILE = 'prompts.jsonl'
GROUND_TRUTH_FILE = 'ground_truths.jsonl'

COMPLETION_LENGTH = 500 
OUTPUT_FILE = 'evabyte_logits.npy'

SEED = 1337
DEVICE = 'cuda'

def load_strong_model(compute_dtype):
    print(f"Loading strong model from {LOCAL_EVABYTE_PATH}...")
    
    if not os.path.exists(LOCAL_EVABYTE_PATH):
        raise FileNotFoundError(f"EvaByte model directory not found at '{LOCAL_EVABYTE_PATH}'")

    tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_EVABYTE_PATH, 
            local_files_only=True,
            trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_EVABYTE_PATH,
        torch_dtype=compute_dtype,
        local_files_only=True,
        trust_remote_code=True
    )

    model.eval()
    model.to(DEVICE)
    print("Strong model loaded successfully.")
    return model, tokenizer

def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32 
   
    print(f'Using device: {DEVICE}')
    print(f'Compute type: {compute_dtype}')

    strong_model, strong_tokenizer = load_strong_model(compute_dtype)
    vocab_size = strong_model.config.vocab_size # Should be 320
    
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"Prompt file not found at '{PROMPT_FILE}'")
    if not os.path.exists(GROUND_TRUTH_FILE):
        raise FileNotFoundError(f"Ground truth file not found at '{GROUND_TRUTH_FILE}'")
        
    with open(PROMPT_FILE, 'r', encoding='utf-8') as p_file:
        prompts = p_file.readlines()
        num_prompts = len(prompts)
        
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as gt_file:
        ground_truths = {json.loads(line)['index']: json.loads(line)['ground_truth'] for line in gt_file}

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
        
        strong_input_ids = strong_tokenizer.encode(prompt_text, return_tensors='pt').to(DEVICE)
        
        with torch.no_grad():
            with ctx:
                for step in range(eval_length):
                    current_length = strong_input_ids.shape[1]
                    position_ids = torch.arange(0, current_length, dtype=torch.long, device=DEVICE).unsqueeze(0)
                    
                    model_output = strong_model(input_ids=strong_input_ids, position_ids=position_ids)
                    logits_tensor = model_output[0][:, -1, :].squeeze()
                    
                    all_logits[i, step, :] = logits_tensor.cpu().numpy()

                    next_token_id = ground_truth_tokens[step]
                    
                    strong_input_ids = torch.cat([strong_input_ids, torch.tensor([[next_token_id]], device=DEVICE)], dim=1)

    print(f"\nSaving logits to '{OUTPUT_FILE}'...")
    np.save(OUTPUT_FILE, all_logits)
    print("Logit extraction complete.")

if __name__ == '__main__':
    main()
