import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "EvaByte/EvaByte"
PROMPT_FILE = "prompts_bos.jsonl"
OUTPUT_FILE = "evabyte_samples_bos.jsonl"
MAX_NEW_TOKENS = 500
TEMPERATURE = 1
TOP_K = None

class RestrictToBytesLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that restricts the vocabulary to the first 256 tokens,
    which correspond to the standard UTF-8 byte values.
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Create a mask that is -inf for all tokens except the first 256
        mask = torch.full_like(scores, -float('inf'))
        mask[:, 64:] = 0
        
        # Apply the mask to the scores (logits)
        scores = scores + mask
        return scores

# -----------------------------------------------------------------------------
# Command-line argument parsing
# -----------------------------------------------------------------------------
def setup_parser():
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(description="Generate text samples from a Hugging Face model.")
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='The name of the Hugging Face model to use.')
    parser.add_argument('--prompt_file', type=str, default=PROMPT_FILE, help='Input file in .jsonl format containing prompts and indices.')
    parser.add_argument('--output_file', type=str, default=OUTPUT_FILE, help='Output file to save the generated samples.')
    parser.add_argument('--max_new_tokens', type=int, default=MAX_NEW_TOKENS, help='Maximum number of new tokens to generate for each prompt.')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='Sampling temperature for generation. Higher is more random.')
    parser.add_argument('--top_k', type=int, default=TOP_K, help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
    return parser

# -----------------------------------------------------------------------------
# Main Generation Logic
# -----------------------------------------------------------------------------
def main():
    """Main function to run the sampling process."""
    parser = setup_parser()
    args = parser.parse_args()

    # --- System Setup ---
    # Check for required files
    if not os.path.exists(args.prompt_file):
        print(f"Error: Prompt file not found at '{args.prompt_file}'")
        print("Please run the create_prompts.py script first.")
        return

    local_model_path = "/home/s2012077/EvaByte_local"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model from: {local_model_path}")
    print("This may take a few minutes as the model is several gigabytes...")

    tokenizer = AutoTokenizer.from_pretrained(
            local_model_path, 
            local_files_only=True,
            trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True
    ).to(device)

    logits_processor = LogitsProcessorList([RestrictToBytesLogitsProcessor()])

    # --- Generation Loop ---
    print(f"Processing prompts from '{args.prompt_file}'...")
    with open(args.prompt_file, 'r', encoding='utf-8') as p_file, \
         open(args.output_file, 'w', encoding='utf-8') as out_file:

        # Read all prompts into memory to show progress
        prompts = p_file.readlines()
        total_prompts = len(prompts)
        
        for i, line in enumerate(prompts):
            print(f"Processing prompt {i+1}/{total_prompts}...")
            try:
                record = json.loads(line)
                start_index = record['index']
                prompt_text = record['prompt']
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  - Warning: Skipping malformed line: {line.strip()} ({e})")
                continue

            # 1. Encode the prompt text to token IDs
            # The tokenizer handles the conversion of the UTF-8 string to byte-level tokens.
            inputs = tokenizer(prompt_text, return_tensors="pt", return_attention_mask=False).to(device)
            input_ids = inputs.input_ids

            # 2. Generate new tokens
            # We use torch.no_grad() to disable gradient calculations for efficiency.
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    logits_processor=logits_processor
                )

            # 3. Decode the generated tokens back to a UTF-8 string
            # We slice the output to remove the original prompt tokens.
            newly_generated_ids = generated_ids[0, input_ids.shape[1]:]
            generated_text = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)

            # 4. Save the result to the output file
            output_record = {
                'index': start_index,
                'generated_text': generated_text
            }
            out_file.write(json.dumps(output_record) + '\n')

            print(f'Written sample {i+1}/{len(prompts)}', flush=True)

    print(f"\nGeneration complete. Samples saved to '{args.output_file}'.")

if __name__ == '__main__':
    main()