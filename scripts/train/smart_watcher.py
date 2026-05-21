#!/usr/bin/env python3
"""
Smart watcher for EEG spatial paper experiments on pi_tpoggio.

Strategy:
  - At each poll, check how many EEG jobs are active (running + pending) in SLURM.
  - If active < MAX_EEG, submit new jobs until we reach MAX_EEG.
  - Uses SLURM job names to detect duplicates (never re-submits the same job name).
  - Saves a done-set to disk so we don't re-submit after restart.

Group GPU cap on pi_tpoggio: ~4 GPUs (shared with rest of lab).
We keep MAX_EEG = 4 to fully use our fair share while leaving room for
lab mates (hatelens, llm-eval, etc. fight for the same pool).
"""
import subprocess, os, time, json, pathlib, argparse

PROJECT   = pathlib.Path("/home/erimoldi/projects/eeg-spatial-paper")
LOG_DIR   = PROJECT / "results" / "slurm_logs"
TEMPLATE  = PROJECT / "scripts" / "train" / "job_template.sh"
DONE_FILE = PROJECT / "results" / "watcher_done.json"

USER      = os.environ.get("USER", "erimoldi")
PARTITION = "pi_tpoggio"
MAX_EEG   = 4   # max EEG jobs active at once on pi_tpoggio

BASE = (
    "--backbone reve "
    "--dataset BNCI2014_001 "
    "--num-classes 4 "
    "--epochs 50 "
    "--batch-size 32 "
)

# ── Short, unambiguous tags ────────────────────────────────────────────────────
SHORT = {
    "none":                 "no",
    "channel_id":           "cid",
    "coords2d":             "c2d",
    "coords3d":             "c3d",
    "coords3d_reference":   "c3r",
    "coords3d_distbias":    "c3b",
    "topology_agnostic":    "top",
}


def build_all_jobs():
    """Return ordered list of all EEG jobs (pilot → core → few-shot → dropout)."""
    jobs = []
    out  = PROJECT / "results"

    # Stage 1: Pilot (9 jobs)
    for v in ["channel_id", "coords2d", "coords3d"]:
        for seed in [42, 123, 456]:
            tag = f"p-{SHORT[v]}-s{seed}"
            jobs.append(dict(
                tag=tag, out=out / "pilot",
                extra=f"--spatial-variant {v} --freeze-policy head_only --seed {seed}"
            ))

    # Stage 2: Core ablation (63 jobs)
    for v in ["none","channel_id","coords2d","coords3d",
              "coords3d_reference","topology_agnostic","coords3d_distbias"]:
        for pol in ["head_only", "partial", "full"]:
            for seed in [42, 123, 456]:
                tag = f"c-{SHORT[v]}-{pol[:4]}-s{seed}"
                jobs.append(dict(
                    tag=tag, out=out / "core_ablation",
                    extra=f"--spatial-variant {v} --freeze-policy {pol} --seed {seed}"
                ))

    # Stage 3: Few-shot (75 jobs)
    for v in ["none","channel_id","coords3d","coords3d_reference","topology_agnostic"]:
        for frac in ["0.01","0.05","0.10","0.25","1.00"]:
            for seed in [42, 123, 456]:
                ftag = frac.replace(".", "p")
                tag = f"f-{SHORT[v]}-{ftag}-s{seed}"
                jobs.append(dict(
                    tag=tag, out=out / "few_shot",
                    extra=(f"--spatial-variant {v} --freeze-policy head_only "
                           f"--label-fraction {frac} --seed {seed}")
                ))

    # Stage 4: Channel dropout (48 jobs)
    for v in ["channel_id","coords3d","coords3d_reference","topology_agnostic"]:
        for rate in ["0.0","0.1","0.3","0.5"]:
            for seed in [42, 123, 456]:
                rtag = rate.replace(".", "p")
                tag = f"d-{SHORT[v]}-{rtag}-s{seed}"
                jobs.append(dict(
                    tag=tag, out=out / "channel_dropout",
                    extra=(f"--spatial-variant {v} --freeze-policy head_only "
                           f"--channel-dropout {rate} --seed {seed}")
                ))

    return jobs


def load_done():
    if DONE_FILE.exists():
        return set(json.loads(DONE_FILE.read_text()))
    return set()


def save_done(done_set):
    DONE_FILE.parent.mkdir(parents=True, exist_ok=True)
    DONE_FILE.write_text(json.dumps(sorted(done_set), indent=2))


def get_active_eeg_names():
    """Return set of EEG job names currently in SLURM (running or pending)."""
    r = subprocess.run(
        ["squeue", "-u", USER, "--partition", PARTITION,
         "--noheader", "--format=%j"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    names = set(r.stdout.decode().strip().split("\n"))
    return {n for n in names if n and n[0] in ("p", "c", "f", "d") and "-" in n}


def count_active_eeg():
    r = subprocess.run(
        ["squeue", "-u", USER, "--partition", PARTITION, "--noheader"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    lines = [l for l in r.stdout.decode().strip().split("\n")
             if l and any(l.split()[0] if l.split() else '' == x for x in ["p","c","f","d"])]
    # Count lines whose jobname starts with p-, c-, f-, d-
    count = 0
    for l in r.stdout.decode().strip().split("\n"):
        parts = l.strip().split()
        if len(parts) >= 3:
            # squeue --noheader format: JOBID PARTITION JOBNAME USER STATE ...
            jname = parts[2]
            if jname and jname[0] in ("p","c","f","d") and "-" in jname:
                count += 1
    return count


def submit(job):
    job["out"].mkdir(parents=True, exist_ok=True)
    train_args = BASE + f"--output-dir {job['out']} " + job["extra"]
    cmd = [
        "sbatch",
        f"--job-name={job['tag']}",
        f"--partition={PARTITION}",
        "--time=06:00:00",
        "--cpus-per-task=8",
        "--mem=64G",
        "--gres=gpu:1",
        f"--output={LOG_DIR}/{job['tag']}_%j.out",
        f"--error={LOG_DIR}/{job['tag']}_%j.err",
        f"--export=ALL,TRAIN_ARGS={train_args}",
        str(TEMPLATE),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = r.stdout.decode().strip()
    stderr = r.stderr.decode().strip()
    if r.returncode == 0:
        return stdout.split()[-1], None
    elif "QOSGrpGRES" in stderr:
        return None, "GRPCAP"
    else:
        return None, stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", type=int, default=180,
                        help="Seconds between checks (default 180)")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    all_jobs = build_all_jobs()
    print(f"Total job definitions: {len(all_jobs)}", flush=True)

    # Load done set
    done = load_done()
    print(f"Already done (from disk): {len(done)}", flush=True)

    # Check SLURM for currently-active EEG jobs (so we don't re-submit)
    active_names = get_active_eeg_names()
    print(f"Currently in SLURM queue: {active_names}", flush=True)
    # Mark manually-submitted pilot jobs as "done" so we skip them
    done |= active_names
    save_done(done)

    pending = [j for j in all_jobs if j["tag"] not in done]
    print(f"Jobs left to submit: {len(pending)}", flush=True)
    print(f"Max simultaneous EEG jobs: {MAX_EEG}", flush=True)
    print(flush=True)

    while pending:
        n_active = count_active_eeg()
        slots = max(0, MAX_EEG - n_active)
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] active={n_active}  slots={slots}  remaining={len(pending)}", flush=True)

        if slots > 0:
            n_sub = 0
            for job in list(pending):
                if n_sub >= slots:
                    break
                jid, err = submit(job)
                if jid:
                    print(f"  OK  {jid}  {job['tag']}", flush=True)
                    done.add(job["tag"])
                    pending.remove(job)
                    n_sub += 1
                    time.sleep(0.3)
                elif err == "GRPCAP":
                    print(f"  group cap — waiting", flush=True)
                    break
                else:
                    print(f"  FAIL {job['tag']}: {err}", flush=True)
            save_done(done)
        else:
            print(f"  no slots — sleeping {args.poll}s", flush=True)

        if pending:
            time.sleep(args.poll)

    print(f"\nAll {len(all_jobs)} jobs submitted!", flush=True)


if __name__ == "__main__":
    main()
