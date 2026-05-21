#!/usr/bin/env python3
import glob
import os
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'eegfm_bench' / 'slurm_logs'
PATTERNS = {
    'baseline': 'efm_cbramod_bcic2a_t2_s*_139845*.out',
    'officialish_mlp': 'efm_cbramod_officialish_mlp_s*_*.out',
    'officialish_linear': 'efm_cbramod_officialish_linear_s*_*.out',
}
METRIC_RE = re.compile(
    r"bcic_2a/(?P<split>eval|test) epoch: (?P<epoch>\d+), "
    r"loss: (?P<loss>[0-9.]+), "
    r"acc: (?P<acc>[0-9.]+), "
    r"balanced_acc: (?P<bacc>[0-9.]+), "
    r"cohen_kappa: (?P<kappa>-?[0-9.]+), "
    r"f1: (?P<f1>[0-9.]+)"
)

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

def best_epoch(rows, metric):
    return max(rows, key=lambda e: (rows[e][metric], rows[e]['acc'], -rows[e]['loss']))

def summarize_file(path):
    evals, tests = parse_metrics(path)
    if not evals or not tests:
        return None
    out = {}
    out['best_eval_bacc_epoch'] = best_epoch(evals, 'balanced_acc')
    out['best_eval_kappa_epoch'] = best_epoch(evals, 'cohen_kappa')
    out['best_test_epoch'] = best_epoch(tests, 'balanced_acc')
    out['final_epoch'] = max(tests)
    out['test_at_best_eval_bacc'] = tests[out['best_eval_bacc_epoch']]['balanced_acc'] * 100.0
    out['test_at_best_eval_kappa'] = tests[out['best_eval_kappa_epoch']]['balanced_acc'] * 100.0
    out['best_test_bacc'] = tests[out['best_test_epoch']]['balanced_acc'] * 100.0
    out['final_test_bacc'] = tests[out['final_epoch']]['balanced_acc'] * 100.0
    return out

def list_files(pattern):
    files = [Path(p) for p in glob.glob(str(LOG_DIR / pattern))]
    def seed_key(p):
        m = re.search(r'_s(\d+)_', p.name)
        return int(m.group(1)) if m else 10**9
    files.sort(key=lambda p: (seed_key(p), p.name))
    return files

def fmt(x):
    return '{:.2f}'.format(x)

def agg(vals):
    if not vals:
        return ''
    if len(vals) == 1:
        return fmt(vals[0])
    return '{} ± {}'.format(fmt(statistics.mean(vals)), fmt(statistics.stdev(vals)))

def main():
    print('| runset | n | test@best-eval-bacc | test@best-eval-kappa | final-test | best-test |')
    print('|---|---:|---:|---:|---:|---:|')
    for label, pattern in PATTERNS.items():
        rows = []
        for path in list_files(pattern):
            data = summarize_file(path)
            if data:
                rows.append(data)
        print('| {} | {} | {} | {} | {} | {} |'.format(
            label,
            len(rows),
            agg([r['test_at_best_eval_bacc'] for r in rows]),
            agg([r['test_at_best_eval_kappa'] for r in rows]),
            agg([r['final_test_bacc'] for r in rows]),
            agg([r['best_test_bacc'] for r in rows]),
        ))

if __name__ == '__main__':
    main()
