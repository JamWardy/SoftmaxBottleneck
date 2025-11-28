import numpy as np
import pandas as pd
import json
import argparse
import itertools
from transformers import AutoTokenizer
from ftfy import fix_text

parser = argparse.ArgumentParser(description="Analyse model logits with various decoding strategies against two ground truths.")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_nucleus_indices(probs, p=0.9):
    sorted_original_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_original_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    nucleus_size = np.searchsorted(cumulative_probs, p) + 1
    nucleus_indices = sorted_original_indices[:nucleus_size]
    return nucleus_indices

def greedy_probs(probabilities):
    greedy_probabilities = np.zeros_like(probabilities)
    greedy_probabilities[np.argmax(probabilities)] = 1.0
    return greedy_probabilities

def top_k_probs(probabilities, k=5):
    if k <= 0: return probabilities
    indices = np.argsort(probabilities)[-k:]
    new_probabilities = np.zeros_like(probabilities)
    new_probabilities[indices] = probabilities[indices]
    prob_sum = np.sum(new_probabilities)
    return new_probabilities / prob_sum if prob_sum > 0 else new_probabilities

def top_p_probs(probabilities, p=0.9):
    if p >= 1.0: return probabilities
    nucleus_indices = get_nucleus_indices(probabilities, p)
    new_probabilities = np.zeros_like(probabilities)
    new_probabilities[nucleus_indices] = probabilities[nucleus_indices]
    prob_sum = np.sum(new_probabilities)
    return new_probabilities / prob_sum if prob_sum > 0 else new_probabilities

def temp_probs(logits, temperature=0.8):
    if temperature <= 0: return softmax(logits)
    return softmax(logits / temperature)

def tto(p_gt, p_bn):
    return np.sum(np.maximum(np.zeros_like(p_gt), p_bn - p_gt))

def tv(p_gt, p_bn):
    return 0.5 * np.sum(np.abs(p_gt - p_bn))

def kl_div(p_gt, p_bn, eps=1e-9):
    return np.sum(p_gt * (np.log(p_gt) - np.log(p_bn)))

def print_results_table(results_dict, total_processed, table_title, show_std_dev=True):
    if total_processed == 0:
        print(f"\nNo data processed for {table_title}. Skipping table.")
        return

    df_data = []
    for i, bn_size in enumerate(BOTTLENECKS):
        for metric in METRICS:
            row_data = {}
            row_data['Metric'] = metric
            row_data['Bottleneck'] = bn_size
            for method in DECODING_METHODS:
                values = results_dict[metric][method][i]
                mean_val = np.mean(values) if values else 0
                
                if show_std_dev:
                    std_val = np.std(values) if values else 0
                    row_data[method] = f"{mean_val:.3f} ± {std_val:.3f}"
                else:
                    row_data[method] = f"{mean_val:.3f}"
            df_data.append(row_data)

    df = pd.DataFrame(df_data)
    df = df.set_index(['Bottleneck', 'Metric'])

    print("\n" + "="*90)
    print(f"      {table_title}")
    print(f"      (Based on {total_processed} data points)")
    print("="*90)
    print(df.to_string())
    
    print("\n" + "-"*90)
    if show_std_dev:
        print("Each cell shows: Mean ± Standard Deviation")
    else:
        print("Each cell shows: Mean")
    print("-" * 90)


BOTTLENECKS = [4, 8, 12, 16, 32, 128, 512]
METRICS = ['TTO', 'NSMD', 'TV', 'KL']
DECODING_METHODS = ['Raw', 'Greedy', 'Top K', 'Top P', 'Temp.']
PROMPT_FILE = 'prompts_bos.jsonl'
GROUND_TRUTH_FILE = 'ground_truths_bos.jsonl'
TOP_P_THRESHOLD = 0.9

print("Loading data...")
with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
    indices = [json.loads(line)['index'] for line in f]
with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
    ground_truths = {json.loads(line)['index']: json.loads(line)['ground_truth'] for line in f}

all_bn_logits = [np.load(f'gpt_logits_{n}_bos.npy') for n in BOTTLENECKS]

ground_truth_logit_files = {
    'EvaByte': 'evabyte_logits_bos_nobos.npy',
    'NanoGPT': 'gpt_logits_trained_gt.npy'
}
all_gt_logits_dict = {name: np.load(path) for name, path in ground_truth_logit_files.items()}
print("Data loaded successfully.")

sample_range = range(len(indices))
byte_offset_range = range(list(all_gt_logits_dict.values())[0].shape[1])
all_pairs = list(itertools.product(sample_range, byte_offset_range))

pairs = all_pairs

all_results = {}
processed_counts = {}

for gt_name, all_gt_logits in all_gt_logits_dict.items():
    print(f"\n--- Processing models against ground truth: {gt_name} ---")
    
    results = {m: {d: [[] for _ in BOTTLENECKS] for d in DECODING_METHODS} for m in METRICS}
    total_processed = 0

    for pair_idx, (sample, byte_offset) in enumerate(pairs):
        index = indices[sample]
        encoded_gt_bytes = ground_truths[index].encode('utf-8')

        if byte_offset >= len(encoded_gt_bytes) or byte_offset >= all_gt_logits.shape[1]:
            continue

        gt_byte_index = encoded_gt_bytes[byte_offset]
        if gt_name == 'NanoGPT':
            gt_logits = all_gt_logits[sample, byte_offset, :]
        else:
            gt_logits = all_gt_logits[sample, byte_offset, 64:]
        gt_probs = softmax(gt_logits)

        nucleus_indices = get_nucleus_indices(gt_probs, p=TOP_P_THRESHOLD)
        tail_indices = np.setdiff1d(np.arange(gt_probs.shape[0]), nucleus_indices)

        for i, bn_logits_set in enumerate(all_bn_logits):
            bn_raw_logits = bn_logits_set[sample, byte_offset, :]
            decoded_probs = {
                'Raw': softmax(bn_raw_logits),
                'Greedy': greedy_probs(softmax(bn_raw_logits)),
                'Top K': top_k_probs(softmax(bn_raw_logits)),
                'Top P': top_p_probs(softmax(bn_raw_logits)),
                'Temp.': temp_probs(bn_raw_logits)
            }
            for method_name, bn_probs in decoded_probs.items():
                results['TTO'][method_name][i].append(tto(gt_probs[tail_indices], bn_probs[tail_indices]))
                results['NSMD'][method_name][i].append(np.sum(gt_probs[nucleus_indices] - bn_probs[nucleus_indices]))
                results['TV'][method_name][i].append(tv(gt_probs, bn_probs))
                results['KL'][method_name][i].append(kl_div(gt_probs, bn_probs))

        total_processed += 1
        if (total_processed % 20000 == 0):
            print(f"  ... processed {total_processed} valid data points for {gt_name}.")
    
    all_results[gt_name] = results
    processed_counts[gt_name] = total_processed
    print(f"Processing for {gt_name} complete. Total valid data points: {total_processed}")

for gt_name, results in all_results.items():
    print_results_table(results, processed_counts[gt_name], f"Metrics Summary (Ground Truth: {gt_name.upper()})", show_std_dev=True)

if len(all_results) == 2:
    combined_results = {m: {d: [[] for _ in BOTTLENECKS] for d in DECODING_METHODS} for m in METRICS}
    eva_results = all_results['EvaByte']
    nano_results = all_results['NanoGPT']

    for metric in METRICS:
        for method in DECODING_METHODS:
            for i in range(len(BOTTLENECKS)):
                combined_values = eva_results[metric][method][i] + nano_results[metric][method][i]
                combined_results[metric][method][i] = combined_values
    
    total_combined_points = processed_counts['EvaByte'] + processed_counts['NanoGPT']
    print_results_table(combined_results, total_combined_points, "Combined Average Metrics", show_std_dev=False)
