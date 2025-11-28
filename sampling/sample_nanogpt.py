import os
import json
import argparse
import torch
from contextlib import nullcontext
from model import GPTConfig, GPT # Assumes model.py is in the same directory

# -----------------------------------------------------------------------------
# Command-line Argument Parsing
# -----------------------------------------------------------------------------
def setup_parser():
    """Sets up and returns the argument parser for the bottleneck size."""
    parser = argparse.ArgumentParser(description="Generate samples from a local GPT model for a specific bottleneck size.")
    parser.add_argument('--n_bottleneck', type=int, required=True, help='The bottleneck size of the model checkpoint to load.')
    parser.add_argument('--top_k', type=int, help='Top-k for sampling.')
    parser.add_argument('--top_p', type=float, help='Top-p for sampling.')
    parser.add_argument('--eta_epsilon', type=float, help='Epsilon for Eta for sampling.')
    return parser

# -----------------------------------------------------------------------------
# Main Generation Logic
# -----------------------------------------------------------------------------
def main():
    """Main function to load model, process prompts, and generate samples."""
    parser = setup_parser()
    args = parser.parse_args()
    n_bottleneck = args.n_bottleneck
    TOP_K = args.top_k
    TOP_P = args.top_p
    ETA_EPSILON = args.eta_epsilon

    OUT_DIR = '../out-fineweb-char/new-16-layer-models' # Base directory for model checkpoints
    PROMPT_FILE = 'prompts_bos.jsonl'               # Input file with prompts
    if (TOP_K is not None):
        OUTPUT_FILE_TEMPLATE = 'local_model_samples_top_k_{TOP_K}_{n_bottleneck}.jsonl'   # Template for the output file name
    elif (TOP_P is not None):
        OUTPUT_FILE_TEMPLATE = 'local_model_samples_top_p_{TOP_P}_{n_bottleneck}.jsonl'
    elif (ETA_EPSILON is not None):
        OUTPUT_FILE_TEMPLATE = 'local_model_samples_eta_{ETA_EPSILON}_{n_bottleneck}.jsonl'
    else:
        OUTPUT_FILE_TEMPLATE = 'local_model_samples_{n_bottleneck}.jsonl'
    NUM_SAMPLES = -1                            # Number of prompts to process (-1 for all)
    MAX_NEW_TOKENS = 500                        # Max tokens to generate per prompt
    TEMPERATURE = 1                          # Sampling temperature
    SEED = 1337
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Initialization and Setup ---
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    if not os.path.exists(PROMPT_FILE):
        print(f"Error: Prompt file not found at '{PROMPT_FILE}'")
        return

    # --- Load Model from Checkpoint ---
    ckpt_path = os.path.join(OUT_DIR, f'ckpt-{n_bottleneck}.pt')
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at '{ckpt_path}'")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # Clean up the state dictionary keys if the model was trained with DDP
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print("Model loaded successfully.")

    # Set up mixed-precision context for CUDA devices
    ctx = nullcontext() if DEVICE == 'cpu' else torch.amp.autocast(device_type=DEVICE, dtype=torch.float16)
    
    # Determine the output file name based on the bottleneck size
    output_file = OUTPUT_FILE_TEMPLATE.format(n_bottleneck=n_bottleneck, TOP_K=TOP_K)

    # --- Generation Loop ---
    print(f"Processing prompts from '{PROMPT_FILE}'...")
    print(f"Generated samples will be saved to '{output_file}'")
    with open(PROMPT_FILE, 'r', encoding='utf-8') as p_file, \
         open(output_file, 'w', encoding='utf-8') as out_file:
        
        prompts = p_file.readlines()
        if NUM_SAMPLES > 0:
            prompts = prompts[:NUM_SAMPLES]
        
        total_prompts = len(prompts)
        print(f"Generating {total_prompts} samples...")

        with torch.no_grad():
            with ctx:
                for i, line in enumerate(prompts):
                    print(f"  - Processing prompt {i+1}/{total_prompts}...")
                    try:
                        record = json.loads(line)
                        index = record['index']
                        prompt_text = record['prompt']
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"    - Warning: Skipping malformed line: {line.strip()} ({e})")
                        continue

                    start_ids = list(prompt_text.encode('utf-8'))
                    x = (torch.tensor(start_ids, dtype=torch.long, device=DEVICE)[None, ...])

                    y = model.generate(x, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, eta_epsilon=ETA_EPSILON)
                    
                    generated_ids = y[0].tolist()
                    newly_generated_ids = generated_ids[len(start_ids):]
                    generated_bytes = bytes(newly_generated_ids)
                    generated_text = generated_bytes.decode('utf-8', errors='ignore')

                    output_record = {'index': index, 'generated_text': generated_text}
                    out_file.write(json.dumps(output_record) + '\n')

    print(f"\nGeneration complete. Samples saved to '{output_file}'.")

if __name__ == '__main__':
    main()
