import numpy as np
import json
import itertools

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

bottlenecks = [4, 8, 12, 16, 32, 128, 512]
PROMPT_FILE = 'prompts_bos.jsonl'
GROUND_TRUTH_FILE = 'ground_truths_bos.jsonl'

with open(PROMPT_FILE, 'r', encoding='utf-8') as p_file:
    indices = [json.loads(line)['index'] for line in p_file]
with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as gt_file:
    ground_truths = {json.loads(line)['index']: json.loads(line)['ground_truth'] for line in gt_file}

all_bn_logits = [np.load(f'gpt_logits_{bn}_bos.npy') for bn in bottlenecks]
gptgt_logits = np.load('gpt_logits_trained_gt.npy')
evabyte_logits = np.load('evabyte_logits_bos_nobos.npy')

pairs = np.array(list(itertools.product(range(100), range(500))))
corrects = [0 for _ in range(len(bottlenecks) + 2)]
counts = 0

for sample, byte_offset_in_gt in pairs:
    index = indices[sample]
    encoded_gt_bytes = ground_truths[index].encode('utf-8')

    if byte_offset_in_gt >= len(encoded_gt_bytes) or byte_offset_in_gt >= evabyte_logits.shape[1]:
        continue

    gt_byte_index = encoded_gt_bytes[byte_offset_in_gt]
    bn_probs = [softmax(all_bn_logits[i][sample, byte_offset_in_gt, :]) for i in range(len(bottlenecks))]
    gptgt_probs = softmax(gptgt_logits[sample, byte_offset_in_gt, :])
    evabyte_probs = softmax(evabyte_logits[sample, byte_offset_in_gt, 64:])

    for i in range(len(bottlenecks)):
        if np.argmax(bn_probs[i]) == gt_byte_index:
            corrects[i] += 1
    if np.argmax(gptgt_probs) == gt_byte_index:
        corrects[-2] += 1
    if np.argmax(evabyte_probs) == gt_byte_index:
        corrects[-1] += 1
    counts += 1

if counts > 0:
    accuracies = np.array(corrects) / counts * 100
    for i, bn in enumerate(bottlenecks):
        print(f"NanoGPT {bn}-BN: {accuracies[i]:.3f}%")
    print(f"NanoGPT 300M: {accuracies[-2]:.3f}%")
    print(f"EvaByte: {accuracies[-1]:.3f}%")
else:
    print("No data points were processed.")
