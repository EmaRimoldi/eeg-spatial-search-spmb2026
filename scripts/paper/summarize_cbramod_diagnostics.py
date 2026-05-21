#!/usr/bin/env python3
import glob
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'eegfm_bench' / 'slurm_logs'
RUNS = [
    ('baseline_seed42', 'efm_cbramod_bcic2a_t2_s42_13984503.out'),
    ('diag_officialish_mlp_seed42', 'efm_cbramod_diag_officialish_mlp_seed42_*.out'),
    ('diag_officialish_linear_seed42', 'efm_cbramod_diag_officialish_linear_seed42_*.out'),
    ('diag_officialish_linear_scratch_seed42', 'efm_cbramod_diag_officialish_linear_scratch_seed42_*.out'),
]
METRIC_RE = re.compile(
    r"bcic_2a/(?P<split>eval|test) epoch: (?P<epoch>\d+), "
    r"loss: (?P<loss>[0-9.]+), "
    r"acc: (?P<acc>[0-9.]+), "
    r"balanced_acc: (?P<bacc>[0-9.]+), "
    r"cohen_kappa: (?P<kappa>-?[0-9.]+), "
    r"f1: (?P<f1>[0-9.]+)"
)

def pick_file(pattern):
    if '*' not in pattern:
        p = LOG_DIR / pattern
        return p if p.exists() else None
    candidates = [Path(p) for p in glob.glob(str(LOG_DIR / pattern))]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return candidates[-1]

def parse_metrics(path):
    evals, tests = {}, {}
    for line in path.read_text().splitlines():
        m = METRIC_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        row = {
            'epoch': int(d['epoch']),
            'loss': float(d['loss']),
            'acc': float(d['acc']),
            'balanced_acc': float(d['bacc']),
            'cohen_kappa': float(d['kappa']),
            'f1': float(d['f1']),
        }
        (evals if d['split'] == 'eval' else tests)[row['epoch']] = row
    return evals, tests

def best_epoch(rows):
    return max(rows, key=lambda e: (rows[e]['balanced_acc'], rows[e]['acc'], -rows[e]['loss']))

def summarize(label, pattern):
    path = pick_file(pattern)
    if path is None:
        return (label, None)
    evals, tests = parse_metrics(path)
    if not evals or not tests:
        return (label, {'file': os.path.relpath(str(path), str(PROJECT_ROOT)), 'status': 'no_metrics'})
    fe = max(tests)
    bee = best_epoch(evals)
    bte = best_epoch(tests)
    return (label, {
        'file': os.path.relpath(str(path), str(PROJECT_ROOT)),
        'final_test_bacc': tests[fe]['balanced_acc'] * 100.0,
        'test_at_best_eval_bacc': tests[bee]['balanced_acc'] * 100.0,
        'best_test_bacc': tests[bte]['balanced_acc'] * 100.0,
        'best_eval_epoch': bee,
        'best_test_epoch': bte,
    })

def main():
    print('| run | test@best-eval B-Acc | best-test B-Acc | final B-Acc | file |')
    print('|---|---:|---:|---:|---|')
    for label, pattern in RUNS:
        _, data = summarize(label, pattern)
        if data is None:
            print('| {} |  |  |  | missing |'.format(label))
            continue
        if data.get('status') == 'no_metrics':
            print('| {} |  |  |  | {} |'.format(label, data['file']))
            continue
        print('| {} | {:.2f} | {:.2f} | {:.2f} | `{}` |'.format(
            label,
            data['test_at_best_eval_bacc'],
            data['best_test_bacc'],
            data['final_test_bacc'],
            data['file'],
        ))

if __name__ == '__main__':
    main()
