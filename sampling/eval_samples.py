import os
import re
import json
import numpy as np
import torch
import nltk
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import mauve
import math
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import words

english_words = set(word.lower() for word in words.words())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sample_files = [
    'local_model_samples_4.jsonl', 
    'local_model_samples_8.jsonl', 
    'local_model_samples_12.jsonl', 
    'local_model_samples_16.jsonl', 
    'local_model_samples_32.jsonl',
    'local_model_samples_128.jsonl',
    'local_model_samples_512.jsonl',
    'local_model_samples_top_k_4.jsonl', 
    'local_model_samples_top_k_8.jsonl', 
    'local_model_samples_top_k_12.jsonl', 
    'local_model_samples_top_k_16.jsonl', 
    'local_model_samples_top_k_32.jsonl',
    'local_model_samples_top_k_128.jsonl',
    'local_model_samples_top_k_512.jsonl',
    'local_model_samples_top_p_4.jsonl', 
    'local_model_samples_top_p_8.jsonl', 
    'local_model_samples_top_p_12.jsonl', 
    'local_model_samples_top_p_16.jsonl', 
    'local_model_samples_top_p_32.jsonl',
    'local_model_samples_top_p_128.jsonl',
    'local_model_samples_top_p_512.jsonl',
    'local_model_samples_4_temp.jsonl', 
    'local_model_samples_8_temp.jsonl', 
    'local_model_samples_12_temp.jsonl', 
    'local_model_samples_16_temp.jsonl', 
    'local_model_samples_32_temp.jsonl',
    'local_model_samples_128_temp.jsonl',
    'local_model_samples_512_temp.jsonl',
    'local_model_samples_greedy_4.jsonl', 
    'local_model_samples_greedy_8.jsonl', 
    'local_model_samples_greedy_12.jsonl', 
    'local_model_samples_greedy_16.jsonl', 
    'local_model_samples_greedy_32.jsonl',
    'local_model_samples_greedy_128.jsonl',
    'local_model_samples_greedy_512.jsonl',
]

def get_strategy(filename):
    if 'greedy' in filename:
        return 'greedy'
    elif 'top_k' in filename:
        return 'top-k'
    elif 'top_p' in filename:
        return 'top-p'
    elif 'temp' in filename:
        return 'temperature'
    else:
        return 'raw'

def get_bottleneck(filename):
    numbers = re.findall(r'(\d+)', filename)
    return numbers[-1] if numbers else 'unknown'

bottleneck_groups = defaultdict(lambda: defaultdict(list))
for f in sample_files:
    bneck = get_bottleneck(f)
    strategy = get_strategy(f)
    bottleneck_groups[bneck][strategy].append(f)

with open('prompts_bos.jsonl', 'r') as pf:
    prompts = {json.loads(l)['index']: json.loads(l)['prompt'] for l in pf}
with open('ground_truths_bos.jsonl', 'r') as gf:
    ground_truths = {json.loads(l)['index']: json.loads(l)['ground_truth'] for l in gf}

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]    = "1"

local_dir = "/work/tc067/tc067/s2012077/models/gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
model     = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True)

def preprocess_completion(prompt: str, completion: str) -> str:
    m_prompt = re.search(r'(\w+)$', prompt)
    m_comp   = re.match (r'^(\w+)', completion)
    if m_prompt and m_comp:
        full_word = m_prompt.group(1) + m_comp.group(1)
        completion = full_word + completion[len(m_comp.group(1)):]

    return completion

def compute_perplexity(texts):
    perps = []
    for j, text in enumerate(texts):
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"]
        with torch.no_grad():
            output = model(input_ids, labels=input_ids)
        loss = output.loss 
        perps.append(torch.exp(loss).item())

    return perps

def compute_repetition(texts):
    ngram_counts = []
    for text in texts:
        tokens = text.split()
        bigrams = zip(tokens, tokens[1:])
        count = Counter(bigrams)
        total = sum(count.values())
        repeat = sum(c for c in count.values() if c > 1)
        if total > 0:
            ngram_counts.append(repeat / total)
    return [np.mean(ngram_counts) if ngram_counts else 0.0]

import mauve.utils

mauve.utils.get_tokenizer = lambda model_name: tokenizer

mauve.utils.get_model     = lambda model_name: model

def compute_mauve(generated_texts, human_texts):
    return [mauve.compute_mauve(
        p_text=generated_texts,
        q_text=human_texts,
        device_id=0 if device=='cuda' else -1,
        verbose=False,
        featurize_model_name=local_dir
    ).mauve]

def total_variation_byte_distribution(human_texts, model_texts):
    counts = np.zeros(256)
    def get_probs(texts):
        for text in texts:
            for b in text.encode('utf-8'):
                counts[b] += 1
        return counts / counts.sum()
    human_probs = get_probs(human_texts)
    model_probs = get_probs(model_texts)
    return [0.5 * np.abs(model_probs - human_probs).sum()]

def proportion_valid_english_words(texts):
    proportions = []
    for text in texts:
        words_in_text = [w.lower() for w in text.split()]
        if not words_in_text:
            continue
        valid = sum(1 for w in words_in_text if w in english_words)
        proportions.append(valid / len(words_in_text))
    return proportions

def average_word_length(texts):
    lengths = []
    for text in texts:
        words_in_text = text.split()
        lengths.append(sum([len(w) for w in words_in_text]) / len(words_in_text))
    return lengths

with open('human_samples.jsonl', 'r') as f:
    human_samples = [json.loads(line)['generated_text'] for line in f]

cleaned_human_samples = []
with open('human_samples.jsonl', 'r') as f:
    for line in f:
        obj = json.loads(line)
        idx  = obj.get('index')
        gen  = obj.get('generated_text')
        if idx is None or gen is None:
            continue
        prompt = prompts[idx]
        cleaned_human_samples.append(preprocess_completion(prompt, gen))

metric_names = ['perplexity', 'repetition', 'mauve', 'byte TV', 'valid words', 'avg word len']
strategies_order = ['raw', 'greedy', 'top-k', 'top-p', 'temperature']

for bneck, strat_files in sorted(bottleneck_groups.items(), key=lambda x: int(x[0])):
    print(f"\nQuality Metrics Table for Bottleneck {bneck}\n")

    metrics_table = defaultdict(lambda: defaultdict(list))
    for strategy, files in strat_files.items():
        all_samples = []
        cleaned_samples = []
        for fname in files:
            with open(fname) as f:
                for line in f:
                    obj = json.loads(line)
                    idx  = obj.get('index')
                    gen  = obj.get('generated_text')
                    if idx is None or gen is None:
                        continue
                    all_samples.append(gen)
                    prompt = prompts[idx]
                    cleaned = preprocess_completion(prompt, gen)
                    cleaned_samples.append(cleaned)
        if not all_samples:
            continue
        metrics_table['perplexity'][strategy] = compute_perplexity(cleaned_samples)
        metrics_table['repetition'][strategy] = compute_repetition(all_samples)
        metrics_table['mauve'][strategy] = compute_mauve(cleaned_human_samples, cleaned_samples)
        metrics_table['byte TV'][strategy] = total_variation_byte_distribution(human_samples, all_samples)
        metrics_table['valid words'][strategy] = proportion_valid_english_words(cleaned_samples)
        metrics_table['avg word len'][strategy] = average_word_length(all_samples)

    for metric in metric_names:
        sample_level_metrics = ['perplexity', 'valid words', 'avg word len']
        for strat in strategies_order:
            scores = metrics_table[metric].get(strat)
            print(f'{metric} - {strat}')
            if (metric in sample_level_metrics):
                mean_val = np.mean(scores)
                std_val = np.std(scores)
                print(f"{mean_val:.12f} ± {std_val:.12f}")
            else:
                mean_val = scores[0]
                print(f"{mean_val:.12f}")
        print()