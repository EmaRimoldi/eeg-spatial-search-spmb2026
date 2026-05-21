#!/usr/bin/env python3
"""
Auto-resubmit remaining experiments as slots free up.

Submits the remaining jobs (few-shot + channel-dropout + any core-ablation
that didn't fit) in batches as the QOS quota permits.

Usage:
    # Immediate attempt (submits up to quota, then exits)
    python scripts/train/resubmit_remaining.py

    # Keep monitoring and resubmitting until all done (--watch mode)
    python scripts/train/resubmit_remaining.py --watch
"""
import subprocess, sys, time, os, json, argparse
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent
PYTHON = "/home/erimoldi/.conda/envs/sparse-hate/bin/python"
LOG_DIR = PROJECT / "results" / "slurm_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PARTITION = "mit_normal_gpu"
TIME = "06:00:00"
BACKBONE = "reve"
DATASET = "BNCI2014_001"
NUM_CLASSES = 4
EPOCHS = 50
BATCH_SIZE = 32

QUEUE_LIMIT = 60   # Leave 4 slots buffer below 64 QOS limit

# ─── All remaining jobs to submit ─────────────────────────────────────────────

def build_job_list():
    jobs = []

    # Core ablation jobs that may not have been submitted yet (topology_agnostic remaining)
    core_dir = PROJECT / "results" / "core_ablation"
    for variant in ["topology_agnostic"]:
        for policy in ["head_only", "partial", "full"]:
            for seed in [42, 123, 456]:
                tag = f"c_{variant[:5]}_{policy[:4]}_s{seed}"
                args = f"--spatial-variant {variant} --freeze-policy {policy} --seed {seed}"
                jobs.append({"tag": tag, "dir": core_dir, "args": args})

    # Stage 3: Few-shot (75 jobs)
    fs_dir = PROJECT / "results" / "few_shot"
    for variant in ["none", "channel_id", "coords3d", "coords3d_reference", "topology_agnostic"]:
        for frac in ["0.01", "0.05", "0.10", "0.25", "1.00"]:
            for seed in [42, 123, 456]:
                ftag = frac.replace(".", "p")
                tag = f"f_{variant[:5]}_{ftag}_s{seed}"
                args = f"--spatial-variant {variant} --freeze-policy head_only --seed {seed} --label-fraction {frac}"
                jobs.append({"tag": tag, "dir": fs_dir, "args": args})

    # Stage 4: Channel dropout (48 jobs)
    cd_dir = PROJECT / "results" / "channel_dropout"
    for variant in ["channel_id", "coords3d", "coords3d_reference", "topology_agnostic"]:
        for rate in ["0.0", "0.1", "0.3", "0.5"]:
            for seed in [42, 123, 456]:
                rtag = rate.replace(".", "p")
                tag = f"d_{variant[:5]}_{rtag}_s{seed}"
                args = f"--spatial-variant {variant} --freeze-policy head_only --seed {seed} --channel-dropout {rate}"
                jobs.append({"tag": tag, "dir": cd_dir, "args": args})

    return jobs


# ─── State tracking ───────────────────────────────────────────────────────────

STATE_FILE = PROJECT / "results" / "resubmit_state.json"

def load_state(all_jobs):
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        return state
    return {"submitted": [], "pending_jobs": [j["tag"] for j in all_jobs]}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ─── SLURM helpers ────────────────────────────────────────────────────────────

def count_running_jobs():
    result = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", "erimoldi"), "--noheader"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    lines = result.stdout.decode().strip().split("\n")
    return len([l for l in lines if l.strip()])

def submit_job(tag, out_dir, extra_args):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrap_cmd = (
        f"source /home/erimoldi/.bashrc 2>/dev/null || true; "
        f"cd {PROJECT}; "
        f"{PYTHON} src/training/train.py "
        f"--backbone {BACKBONE} --dataset {DATASET} "
        f"--num-classes {NUM_CLASSES} --epochs {EPOCHS} "
        f"--batch-size {BATCH_SIZE} --output-dir {out_dir} {extra_args}"
    )

    cmd = [
        "sbatch",
        f"--job-name={tag}",
        f"--partition={PARTITION}",
        f"--time={TIME}",
        "--cpus-per-task=8",
        "--mem=32G",
        "--gres=gpu:1",
        f"--output={LOG_DIR}/{tag}_%j.out",
        f"--error={LOG_DIR}/{tag}_%j.err",
        f"--wrap={wrap_cmd}",
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = result.stdout.decode().strip()
    stderr = result.stderr.decode().strip()

    if result.returncode == 0:
        job_id = stdout.split()[-1]
        return job_id, None
    elif "QOSMaxSubmitJobPerUserLimit" in stderr:
        return None, "QUOTA"
    else:
        return None, stderr


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true",
                        help="Keep running until all jobs submitted")
    parser.add_argument("--poll-interval", type=int, default=300,
                        help="Seconds between queue checks in --watch mode")
    args = parser.parse_args()

    all_jobs = build_job_list()
    state = load_state(all_jobs)

    pending_tags = set(state["pending_jobs"])
    pending_jobs = [j for j in all_jobs if j["tag"] in pending_tags]

    if not pending_jobs:
        print(f"All {len(all_jobs)} remaining jobs already submitted.")
        return

    print(f"Remaining jobs to submit: {len(pending_jobs)}")

    def submit_batch():
        n_current = count_running_jobs()
        slots_available = max(0, QUEUE_LIMIT - n_current)
        print(f"Queue: {n_current} running | {slots_available} slots available")

        if slots_available == 0:
            return 0

        n_submitted = 0
        for job in list(pending_jobs):
            if n_submitted >= slots_available:
                break
            job_id, err = submit_job(job["tag"], job["dir"], job["args"])
            if err == "QUOTA":
                print(f"  Quota hit after {n_submitted} submissions")
                break
            elif job_id:
                print(f"  OK {job_id}  {job['tag']}")
                state["submitted"].append({"tag": job["tag"], "job_id": job_id})
                pending_tags.discard(job["tag"])
                pending_jobs.remove(job)
                n_submitted += 1
            else:
                print(f"  FAIL {job['tag']}: {err}")
            time.sleep(0.1)

        state["pending_jobs"] = list(pending_tags)
        save_state(state)
        return n_submitted

    submitted_now = submit_batch()
    print(f"\nSubmitted {submitted_now} jobs this run.")
    print(f"Remaining: {len(pending_jobs)}")

    if not args.watch:
        if pending_jobs:
            print(f"\nRe-run this script later to submit the remaining {len(pending_jobs)} jobs:")
            print(f"  python {Path(__file__).relative_to(PROJECT)} --watch")
        return

    # Watch mode: keep polling until all submitted
    while pending_jobs:
        print(f"\nSleeping {args.poll_interval}s... ({len(pending_jobs)} jobs remaining)")
        time.sleep(args.poll_interval)
        submitted_now = submit_batch()
        print(f"Submitted {submitted_now} more jobs. Total remaining: {len(pending_jobs)}")

    print(f"\nAll jobs submitted! Total: {len(state['submitted'])}")


if __name__ == "__main__":
    main()
