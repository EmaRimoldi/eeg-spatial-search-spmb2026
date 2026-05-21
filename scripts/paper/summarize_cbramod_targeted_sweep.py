#!/usr/bin/env python3
import glob
import os
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'eegfm_bench' / 'slurm_logs'
PATTERNS = {
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

def best_epoch(rows, metric='balanced_acc'):
    return max(rows, key=lambda e: (rows[e][metric], rows[e]['acc'], -rows[e]['loss']))

def list_files(pattern):
    files = [Path(p) for p in glob.glob(str(LOG_DIR / pattern))]
    def seed_key(p):
        m = re.search(r'_s(\d+)_', p.name)
        return int(m.group(1)) if m else 10**9
    files.sort(key=lambda p: (seed_key(p), p.name))
    return files

def summarize(path):
    evals, tests = parse_metrics(path)
    if not evals or not tests:
        return None
    bee = best_epoch(evals, 'balanced_acc')
    bek = best_epoch(evals, 'cohen_kappa')
    bte = best_epoch(tests, 'balanced_acc')
    fe = max(tests)
    return {
        'file': os.path.relpath(str(path), str(PROJECT_ROOT)),
        'seed': int(re.search(r'_s(\d+)_', path.name).group(1)),
        'test_at_best_eval_bacc': tests[bee]['balanced_acc'] * 100.0,
        'test_at_best_eval_kappa': tests[bek]['balanced_acc'] * 100.0,
        'best_test_bacc': tests[bte]['balanced_acc'] * 100.0,
        'final_test_bacc': tests[fe]['balanced_acc'] * 100.0,
    }

def fmt(x):
    return '{:.2f}'.format(x)

def meanstd(vals):
    if not vals:
        return ''
    if len(vals) == 1:
        return fmt(vals[0])
    return '{} ± {}'.format(fmt(statistics.mean(vals)), fmt(statistics.stdev(vals)))

def main():
    print('| variant | n | test@best-eval-bacc | test@best-eval-kappa | final-test | best-test |')
    print('|---|---:|---:|---:|---:|---:|')
    for label, pattern in PATTERNS.items():
        rows = [summarize(p) for p in list_files(pattern)]
        rows = [r for r in rows if r is not None]
        print('| {} | {} | {} | {} | {} | {} |'.format(
            label,
            len(rows),
            meanstd([r['test_at_best_eval_bacc'] for r in rows]),
            meanstd([r['test_at_best_eval_kappa'] for r in rows]),
            meanstd([r['final_test_bacc'] for r in rows]),
            meanstd([r['best_test_bacc'] for r in rows]),
        ))

if __name__ == '__main__':
    main()
