import json
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import Counter, defaultdict
import re
import nltk
from nltk.corpus import words
import seaborn as sns
from transformers import AutoTokenizer
import string

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

PROMPT_FILE = 'prompts_bos.jsonl'
GROUND_TRUTH_FILE = 'ground_truths_bos.jsonl'

with open(PROMPT_FILE, 'r', encoding='utf-8') as p_file:
    prompts = {json.loads(line)['index']: json.loads(line)['prompt'] for line in p_file}
    
with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as gt_file:
    ground_truths = {json.loads(line)['index']: json.loads(line)['ground_truth'] for line in gt_file}

indices = list(prompts.keys())

english_words = set(word.lower() for word in words.words())

files_to_evaluate = [
    'local_model_samples_4.jsonl', 
    'local_model_samples_8.jsonl', 
    'local_model_samples_12.jsonl', 
    'local_model_samples_16.jsonl', 
    'local_model_samples_32.jsonl',
    'local_model_samples_128.jsonl',
    'local_model_samples_512.jsonl',
    'evabyte_samples_bos.jsonl',
    'human_samples.jsonl'
]

def get_byte_frequencies(text):
    byte_counts = Counter()
    for char in text:
        for byte in char.encode('utf-8'):
            byte_counts[byte] += 1
    return byte_counts

def get_words_from_combined_text(prompt, generated_text):
    full_text = prompt + generated_text
    
    all_words = re.findall(r'\b[a-zA-Z]+\b', full_text.lower())
    
    prompt_words = re.findall(r'\b[a-zA-Z]+\b', prompt.lower())
    
    generated_words = re.findall(r'\b[a-zA-Z]+\b', generated_text.lower())
    
    if not all_words or len(all_words) <= len(prompt_words):
        return []
    
    generated_portion_words = all_words[len(prompt_words):]
    
    ends_mid_word = (generated_text and 
                     generated_text[-1].isalpha() and 
                     len(generated_portion_words) > 0)
    
    if ends_mid_word:
        generated_portion_words = generated_portion_words[:-1]
    
    return generated_portion_words

def calculate_word_metrics(prompt, generated_text):
    words = get_words_from_combined_text(prompt, generated_text)
    
    if not words:
        return 0, 0
    
    avg_length = sum(len(word) for word in words) / len(words)
    
    valid_words = sum(1 for word in words if word in english_words)
    valid_proportion = valid_words / len(words)
    
    return avg_length, valid_proportion

print("--- Starting Extended Analysis ---")

model_byte_frequencies = {}
model_word_lengths = {}
model_english_proportions = {}

for sample_file in files_to_evaluate:
    model_name = sample_file.replace('.jsonl', '').replace('local_model_samples_', '').replace('_samples', '')
    
    try:
        with open(sample_file, 'r', encoding='utf-8') as s_file:
            samples = {json.loads(line)['index']: json.loads(line)['generated_text'] for line in s_file}
    except FileNotFoundError:
        print(f"Warning: File not found: {sample_file}. Skipping.")
        continue

    print(f"Processing {model_name}...")
    
    combined_byte_freq = Counter()
    word_lengths = []
    english_proportions = []
    
    num_samples = 0
    
    for index in indices:
        if index in samples and index in prompts and index in ground_truths:
            num_samples += 1
            
            prompt_text = prompts[index]
            generated_text = samples[index]
            
            byte_freq = get_byte_frequencies(generated_text)
            combined_byte_freq.update(byte_freq)
            
            avg_length, valid_prop = calculate_word_metrics(prompt_text, generated_text)
            if avg_length > 0:
                word_lengths.append(avg_length)
                english_proportions.append(valid_prop)
    
    if num_samples > 0:
        model_byte_frequencies[model_name] = combined_byte_freq
        model_word_lengths[model_name] = word_lengths
        model_english_proportions[model_name] = english_proportions
        
        print(f"  Processed {num_samples} samples")
        print(f"  Average word length: {np.mean(word_lengths):.3f}")
        print(f"  Average English word proportion: {np.mean(english_proportions):.3f}")

model_relative_freqs = {}
for model, byte_freqs in model_byte_frequencies.items():
    total = sum(byte_freqs.values())
    if total > 0:
        rel_freqs = {b: count / total for b, count in byte_freqs.items()}
    else:
        rel_freqs = {b: 0 for b in range(256)}
    model_relative_freqs[model] = rel_freqs

if 'human' not in model_relative_freqs:
    print("Error: No human sample frequencies found. Cannot compute TVD.")
    tvd_scores = {}
else:
    human_freqs = model_relative_freqs['human']

    tvd_scores = {}
    for model, freqs in model_relative_freqs.items():
        tvd = 0.5 * sum(abs(freqs.get(b, 0) - human_freqs.get(b, 0)) for b in range(256))
        tvd_scores[model] = tvd

print("\n--- SUMMARY METRICS BY MODEL ---")
print(f"{'Model':<20} {'Avg Word Len':<15} {'Eng Word Prop':<18} {'TVD (vs Human)':<15}")
print("-" * 70)

model_names = sorted(model_word_lengths.keys())
for name in model_names:
    avg_len = np.mean(model_word_lengths[name]) if model_word_lengths[name] else float('nan')
    avg_eng = np.mean(model_english_proportions[name]) if model_english_proportions[name] else float('nan')
    tvd = tvd_scores.get(name, float('nan'))

    print(f"{name:<20} {avg_len:<15.3f} {avg_eng:<18.3f} {tvd:<15.4f}")

def generate_labels_with_tokenizer(tokenizer):
    """
    Generates a list of 256 strings for byte labels using the tokenizer.
    """
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

tokenizer = AutoTokenizer.from_pretrained("../EvaByte_local", local_files_only=True, trust_remote_code=True)
byte_labels = generate_labels_with_tokenizer(tokenizer)

def plot_relative_byte_frequency_histograms(model_relative_freqs):
    model_names = ['4', '8', '12', '32', '512']
    num_models = len(model_names)
    num_cols = 3
    num_rows = math.ceil(num_models / num_cols)

    human_freqs = model_relative_freqs.get('human', {})
    max_y = 0
    for freqs in model_relative_freqs.values():
        model_max = max(freqs.get(b, 0) for b in range(256))
        if model_max > max_y:
            max_y = model_max

    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6), constrained_layout=True)
    axes = axes.flatten()

    tick_spacing = 16
    tick_locations = np.arange(0, 256, tick_spacing)
    tick_labels = [byte_labels[i] for i in tick_locations]

    w = 0.35
    offset = w * 1.1

    for i, model in enumerate(model_names):
        ax = axes[i]
        freqs = model_relative_freqs[model]
        byte_values = list(range(256))
        rel_frequencies = [freqs.get(b, 0) for b in byte_values]
        ax.bar(byte_values, rel_frequencies, width=1.0)

        ax.set_title(f"Nano-GPT {model}-BN")

        ax.set_ylim(0, max_y * 1.1)
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels, rotation=90)
        if (i > 2):
            ax.set_xlabel("Byte")
        else:
            ax.set_xlabel("")
        if (i == 0 or i == 3):
            ax.set_ylabel('Rel. Freq.')

    byte_values = list(range(256))
    rel_frequencies = [human_freqs.get(b, 0) for b in byte_values]
    axes[-1].bar(byte_values, rel_frequencies, width=1.0)
    axes[-1].set_ylim(0, max_y * 1.1)
    axes[-1].set_title("Human")
    axes[-1].set_xticks(tick_locations)
    axes[-1].set_xticklabels(tick_labels, rotation=90)
    axes[-1].set_xlabel("Byte")

    plt.suptitle('Byte Relative Frequency Histograms')
    plt.show()

plot_relative_byte_frequency_histograms(model_relative_freqs)