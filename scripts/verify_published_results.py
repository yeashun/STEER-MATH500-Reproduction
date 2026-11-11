"""Recompute published statistics from compact records; standard library only."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]

def read_csv(path):
    with (ROOT / path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def close(actual, expected, message):
    require(abs(actual - expected) < 1e-9, f'{message}: {actual} != {expected}')

def main():
    records = read_csv('reports/main/score_token_counts.csv')
    require(len(records) == 6000, 'Expected 12 runs of 500 questions')
    groups = defaultdict(list)
    for row in records:
        groups[row['method'], int(row['seed'])].append(row)
    require(set(groups) == {(method, seed) for method in ['small','large','steer'] for seed in range(4)}, 'Unexpected runs')
    expected = json.loads((ROOT / 'reports/main/summary.json').read_text(encoding='utf-8'))
    runs = {}
    for (method, seed), rows in groups.items():
        require(len(rows) == 500 and {int(r['idx']) for r in rows} == set(range(500)), f'Incomplete {method}/{seed}')
        require(all(r['correct'] in {'0','1'} for r in rows), 'Invalid score')
        require(all(int(r[k]) >= 0 for r in rows for k in ['small_used_tokens','small_discarded_tokens','large_generated_tokens']), 'Negative token count')
        accuracy = 100 * sum(int(r['correct']) for r in rows) / len(rows)
        flops = sum(3e9 * (int(r['small_used_tokens']) + int(r['small_discarded_tokens'])) + 14e9 * int(r['large_generated_tokens']) for r in rows) / len(rows) / 1e12
        runs[method, seed] = (accuracy, flops)
    for method in ['small','large','steer']:
        accuracy, flops = zip(*(runs[method, seed] for seed in range(4)))
        for values, key in [(accuracy, 'accuracy_pct'), (flops, 'mean_flops_t')]:
            close(mean(values), expected[method][key]['mean'], f'{method}/{key}/mean')
            close(stdev(values), expected[method][key]['sample_sd'], f'{method}/{key}/sd')
    close(100 * (1 - expected['steer']['mean_flops_t']['mean'] / expected['large']['mean_flops_t']['mean']),
          expected['steer']['saving_from_ratio_of_mean_flops_pct'], 'FLOPs saving')
    for row in read_csv('reports/main/per_seed.csv'):
        method, seed = row['method'], int(row['seed'])
        close(float(row['rule_accuracy']), runs[method, seed][0], 'per-seed accuracy')
        close(float(row['mean_flops_t']), runs[method, seed][1], 'per-seed cost')

    pairs = read_csv('reports/experiment1/per_question.csv')
    require(len(pairs) == 2000, 'Expected 2000 model pairs')
    scores = {(r['method'], int(r['seed']), int(r['idx'])): int(r['correct']) for r in records}
    require(len({(int(r['seed']), int(r['idx'])) for r in pairs}) == 2000, 'Duplicate pair')
    states = Counter()
    histories = defaultdict(list)
    for row in pairs:
        seed, idx = int(row['seed']), int(row['idx'])
        state = f"{scores['small', seed, idx]}{scores['large', seed, idx]}"
        require(row['state'] == state, f'Pair score mismatch: {seed}/{idx}')
        states[state] += 1
        histories[idx].append(state)
    require(states == Counter({'11':1402,'01':228,'10':85,'00':285}), 'Four-state counts differ')
    categories = Counter(next(iter(set(seq))) if len(set(seq)) == 1 else 'variable' for seq in histories.values())
    require(categories == Counter({'11':288,'01':10,'10':2,'00':41,'variable':159}), 'State stability differs')

    selected = read_csv('reports/experiment2/selected_01_00.csv')
    require(len(selected) == 513, 'Expected 513 eligible records')
    pair_lookup = {(int(r['seed']), int(r['idx'])): r for r in pairs}
    positives, negatives = [], []
    for row in selected:
        seed, idx = int(row['seed']), int(row['idx'])
        original = pair_lookup[seed, idx]
        require(row['state'] == original['state'] and row['state'] in {'01','00'}, 'Proxy label mismatch')
        current = ''.join(original['small_pred'].split())
        require(bool(current), 'Unexpected missing prediction')
        matches = sum(current == ''.join(pair_lookup[other,idx]['small_pred'].split()) for other in range(4) if other != seed)
        close(float(row['answer_agreement_proxy']), matches / 3, 'Proxy recomputation')
        (positives if row['state'] == '01' else negatives).append(matches / 3)
    auc = sum((p > n) + .5 * (p == n) for p in positives for n in negatives) / len(positives) / len(negatives)
    proxy = json.loads((ROOT / 'reports/experiment2/summary.json').read_text(encoding='utf-8'))['confidence_proxy_analysis']
    close(auc, proxy['auc_higher_agreement_predicts_01'], 'AUROC')
    close(1 - auc, proxy['auc_lower_agreement_predicts_01'], 'Reverse AUROC')
    print('PASS: 12 runs / 6000 score-token records; means, SDs, FLOPs saving;')
    print('      2000 model pairs; 500 question histories; 513 agreement proxies and AUROC.')
    print('Scope: saved scores and statistics only; no model generation or answer regrading.')

if __name__ == '__main__':
    main()
