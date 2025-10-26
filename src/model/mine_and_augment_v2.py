# src/model/mine_and_augment_v2.py
import os
import sys
import json
import time
import subprocess
import argparse
import re
from pathlib import Path

import pandas as pd

# ======================================================================================
# Paths & helpers
# ======================================================================================

def _cwd() -> Path:
    """Return the directory that contains this file (src/model)."""
    return Path(__file__).resolve().parent

def _project_root() -> Path:
    """Assume repo layout .../<project_root>/src/model/<this_file>."""
    return _cwd().parents[1]  # src

def _repo_root() -> Path:
    """One level above src => project root."""
    return _cwd().parents[1]

def _data_dir() -> Path:
    return _repo_root() / "data"

def _processed_dir() -> Path:
    return _data_dir() / "processed"

def _models_dir() -> Path:
    return _repo_root() / "models"

def _ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _read_csv_safe(path: Path, required_cols=None) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame(columns=required_cols or [])
    df = pd.read_csv(path)
    if required_cols:
        for c in required_cols:
            if c not in df.columns:
                df[c] = []
        df = df[required_cols]
    return df

# Regex to capture Validation(calibrated) F1 from run_discriminator stdout (optional)
VAL_CAL_RE = re.compile(
    r"Validation \(calibrated\):\s*\{[^}]*['\"]f1['\"]:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE
)

# ======================================================================================
# Step 1: Train discriminator + mine hard examples
# ======================================================================================

def run_discriminator(
    backbone: str,
    iteration: int,
    train_base: str,
    train_aug: str,
    # passthrough training knobs (optional)
    epochs: int | None = None,
    lr: float | None = None,
    bsz_train: int | None = None,
    bsz_eval: int | None = None,
    max_len: int | None = None,
    rdrop_alpha: float | None = None,
    label_smoothing: float | None = None,
    target_recall_spam: float | None = None,
    mine_topk: int | None = None,
) -> tuple[float, dict]:
    """
    Launch run_discriminator.py with correct flags, then collect:
      - Validation F1 (from metrics.json, fallback to parsed stdout)
      - Hard-mined files => combined hard_examples.csv
      - Counts dict for logging/early-stop
    """
    print(f"\n🚀 Iteration {iteration}: Training discriminator...")
    output_name = f"outputs/discriminator_iter{iteration:02d}"

    cmd = [
        sys.executable,
        "run_discriminator.py",
        "--backbone", backbone,
        "--train_base_file", train_base,
        "--train_aug_file",  train_aug,
        "--output_name",     output_name,
    ]

    # Optional pass-throughs
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if lr is not None:
        cmd += ["--lr", str(lr)]
    if bsz_train is not None:
        cmd += ["--bsz_train", str(bsz_train)]
    if bsz_eval is not None:
        cmd += ["--bsz_eval", str(bsz_eval)]
    if max_len is not None:
        cmd += ["--max_len", str(max_len)]
    if rdrop_alpha is not None:
        cmd += ["--rdrop_alpha", str(rdrop_alpha)]
    if label_smoothing is not None:
        cmd += ["--label_smoothing", str(label_smoothing)]
    if target_recall_spam is not None:
        cmd += ["--target_recall_spam", str(target_recall_spam)]
    if mine_topk is not None:
        cmd += ["--mine_topk", str(mine_topk)]

    # -------------- launch (streamed) --------------
    proc = subprocess.Popen(
        cmd,
        cwd=_cwd(),
        text=True,
        stdout=None,  # inherit parent's stdout
        stderr=None  # inherit parent's stderr; tqdm writes here
    )
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)

    # -------------- read metrics.json (no stdout fallback) --------------
    run_dir = _models_dir() / output_name
    metrics_path = run_dir / "metrics.json"
    val_f1 = 0.0
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            val_f1 = float(m.get("val_metrics", {}).get("f1", m.get("f1", 0.0)))
            print(f"✅ Validation F1 (from metrics.json): {val_f1:.4f}")
        except Exception as e:
            print(f"⚠️ Failed to parse metrics.json @ {metrics_path}: {e}")
    # else: no stdout regex parsing here because we didn't capture output

    # -------------- gather mined outputs --------------
    TARGET_POOL_SIZE = 100  # Your desired hard cap

    # Define paths to the candidate files
    fn_file = _processed_dir() / "hard_spam_false_negatives.csv"
    borderline_file = _processed_dir() / "hard_spam_borderline.csv"
    bottompct_file = _processed_dir() / "hard_spam_bottompct.csv"
    ood_high_file = _processed_dir() / "hard_ood_high.csv"

    # Load all potential candidates
    df_fn = _read_csv_safe(fn_file, ["label", "message"])
    df_borderline = _read_csv_safe(borderline_file, ["label", "message"])
    df_bottom = _read_csv_safe(bottompct_file, ["label", "message"])
    df_ood = _read_csv_safe(ood_high_file, ["label", "message"])
    # Ensure we only consider OOD examples that are actually spam
    df_ood = df_ood[df_ood["label"] == 1]

    # --- Prioritized Pool Creation ---
    # We will fill our pool of TARGET_POOL_SIZE, prioritizing the most valuable examples.

    final_pool_df = pd.DataFrame(columns=["label", "message"])
    seen_messages = set()

    # Helper function to add unique messages from a dataframe to the pool
    def add_to_pool(pool_df, seen_set, candidates_df, budget):
        # Exclude messages we've already added
        unique_candidates = candidates_df[~candidates_df["message"].isin(seen_set)]
        # Limit to the remaining budget
        to_add = unique_candidates.head(budget)
        # Update the pool and the set of seen messages
        new_pool = pd.concat([pool_df, to_add], ignore_index=True)
        seen_set.update(to_add["message"])
        return new_pool, seen_set

    # 1. Highest Priority: False Negatives (take all available, up to the cap)
    remaining_budget = TARGET_POOL_SIZE - len(final_pool_df)
    if remaining_budget > 0 and not df_fn.empty:
        final_pool_df, seen_messages = add_to_pool(final_pool_df, seen_messages, df_fn, remaining_budget)
        print(f"  Added {len(df_fn)} FNs. Pool size: {len(final_pool_df)}")

    # 2. High Priority: OOD Spam
    remaining_budget = TARGET_POOL_SIZE - len(final_pool_df)
    if remaining_budget > 0 and not df_ood.empty:
        ood_added_count = min(remaining_budget, len(df_ood[~df_ood['message'].isin(seen_messages)]))
        final_pool_df, seen_messages = add_to_pool(final_pool_df, seen_messages, df_ood, remaining_budget)
        print(f"  Added {ood_added_count} OOD examples. Pool size: {len(final_pool_df)}")

    # 3. Medium Priority: Borderline Spam
    remaining_budget = TARGET_POOL_SIZE - len(final_pool_df)
    if remaining_budget > 0 and not df_borderline.empty:
        borderline_added_count = min(remaining_budget,
                                     len(df_borderline[~df_borderline['message'].isin(seen_messages)]))
        final_pool_df, seen_messages = add_to_pool(final_pool_df, seen_messages, df_borderline, remaining_budget)
        print(f"  Added {borderline_added_count} borderline examples. Pool size: {len(final_pool_df)}")

    # 4. Lower Priority: Top-Energy Spam (to fill the rest)
    remaining_budget = TARGET_POOL_SIZE - len(final_pool_df)
    if remaining_budget > 0 and not df_bottom.empty:
        bottom_added_count = min(remaining_budget, len(df_bottom[~df_bottom['message'].isin(seen_messages)]))
        final_pool_df, seen_messages = add_to_pool(final_pool_df, seen_messages, df_bottom, remaining_budget)
        print(f"  Added {bottom_added_count} top-energy examples. Pool size: {len(final_pool_df)}")

    # Final de-duplication just in case
    final_pool_df = final_pool_df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    # Update counts for logging
    counts = {
        "false_negatives": len(df_fn),
        "borderline": len(df_borderline),
        "bottom_p": len(df_bottom),
        "ood_high": len(df_ood),
        "final_pool_size": len(final_pool_df)
    }

    combined_path = _processed_dir() / "hard_examples.csv"

    # ✅ ALWAYS write the merged and capped pool
    _ensure_parent(combined_path)
    final_pool_df.to_csv(combined_path, index=False, encoding="utf-8")

    print("\nHARD MINING & PRIORITIZATION DONE:")
    print(f"  Candidate FNs: {counts['false_negatives']}")
    print(f"  Candidate Borderline: {counts['borderline']}")
    print(f"  Candidate Top-Energy: {counts['bottom_p']}")
    print(f"  Candidate OOD Spam: {counts['ood_high']}")
    print(f"✅ Final prioritized pool of {len(final_pool_df)} examples saved to {combined_path}")

    if final_pool_df.empty:
        print("⚠️ Hard pool is empty this iteration.")

    return val_f1, counts

# ======================================================================================
# Step 2: LLM data generator
# ======================================================================================

def run_generator(iter_idx: int, num_msgs: int, mode: str = "hard", hard_file: Path | None = None) -> str:
    """
    Launch llm_generator_v2.py in either `hard` or `full` mode.
    - If mode == 'hard' but hard_file missing/empty, automatically fallback to 'full'.
    Returns the path to the generated CSV.
    """
    out = _data_dir() / f"augmented_spam_iter{iter_idx:02d}.csv"
    _ensure_parent(out)

    # If user asked for hard but there’s no pool, go to full
    if mode == "hard":
        if not hard_file or not hard_file.exists():
            print("ℹ️ No hard pool available; switching generator to 'full' mode for this iteration.")
            mode = "full"
        else:
            dfh = _read_csv_safe(hard_file, ["message", "label"])
            if dfh.empty:
                print("ℹ️ Hard pool file is empty; switching generator to 'full' mode for this iteration.")
                mode = "full"

    # Build command
    base = [
        sys.executable,
        "llm_generator_v2.py",
        "--mode", mode,
        "--num_messages", str(num_msgs),
        "--output", str(out),
        "--batch_size", "16",
        "--max_new_tokens", "32",
    ]

    if mode == "hard":
        # Use mined pool and dedup against current train_aug
        base += [
            "--hard_files", str(hard_file),
            "--dedup_train"
        ]
        # Some sane prompt/few-shot defaults for hard conditioning (your script supports these)
        base += [
            "--hard_prompts", "12",
            "--hard_few", "3",
            "--per_prompt", "8",
        ]

    print(f"✍️ Iteration {iter_idx}: Generating augmented spam ({mode})...")
    subprocess.run(base, check=True, cwd=_cwd())
    print(f"✅ Generated {num_msgs} messages in {mode} mode → saved to {out}")
    return str(out)


# ======================================================================================
# Step 3: Merge new augmented data into cumulative train_aug
# ======================================================================================

def merge_augmented_into_train(new_aug_path: str, current_train_aug_path: str) -> str:
    """
    Append new augmented spam (label=1) into data/processed/train_sms_mistral_augmented.csv
    with de-duplication by message text.
    Returns the path to the merged CSV.
    """
    out_path = _processed_dir() / "train_sms_mistral_augmented.csv"

    base = Path(current_train_aug_path) if current_train_aug_path else None
    df_base = _read_csv_safe(base, ["message", "label"]) if base and base.exists() else pd.DataFrame(columns=["message", "label"])
    df_new = _read_csv_safe(Path(new_aug_path), ["message", "label"])
    if "label" not in df_new.columns:
        df_new["label"] = 1

    merged = pd.concat([df_base, df_new], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["message"]).reset_index(drop=True)
    after = len(merged)
    print(f"[base_dedup] before={before} | after={after}")

    _ensure_parent(out_path)
    merged.to_csv(out_path, index=False, encoding="utf-8")
    return str(out_path)


# ======================================================================================
# Pipeline (mine → generate → merge) with early stopping
# ======================================================================================

def run_pipeline(
    iters: int,
    per_iter_new: int,
    patience: int,
    backbone: str,
    train_base: str,
    train_aug: str,
    min_iters: int = 1,
    no_early_stop: bool = False,
    reset_summary: bool = False,
    exp_id: str | None = None,
    # passthrough knobs to run_discriminator
    epochs: int | None = None,
    lr: float | None = None,
    bsz_train: int | None = None,
    bsz_eval: int | None = None,
    max_len: int | None = None,
    rdrop_alpha: float | None = None,
    label_smoothing: float | None = None,
    target_recall_spam: float | None = None,
    mine_topk: int | None = None,
):
    """
    Main loop:
      1) Train discriminator, mine hard examples (+ OOD-high)
      2) Generate new spam with LLM (hard if pool exists, else full)
      3) Merge into train_aug and continue
    """
    data_dir = _data_dir()
    summary_path = data_dir / "mine_and_augment_summary.json"

    # Fresh per-run state (don’t carry over a previous run’s “best” unless resuming intentionally).
    state = {
        "exp_id": exp_id or time.strftime("%Y%m%d-%H%M%S"),
        "backbone": backbone,
        "global_best_f1_this_run": None,
        "no_improve": 0,
        "history": [],
    }

    if summary_path.exists() and not reset_summary:
        try:
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            if prev.get("backbone") == backbone and prev.get("exp_id"):
                state["exp_id"] = prev["exp_id"]
                print(f"ℹ️ Continuing with exp_id={state['exp_id']} (fresh best this run).")
        except Exception:
            pass
    else:
        print("ℹ️ Resetting summary (fresh run).")

    _ensure_parent(summary_path).write_text(json.dumps(state, indent=2), encoding="utf-8")

    # Loop
    for it in range(1, iters + 1):
        print(f"\n=== Iteration {it}/{iters} ===\n")

        # 1) Train discriminator + mine
        val_f1, counts = run_discriminator(
            backbone=backbone,
            iteration=it,
            train_base=train_base,
            train_aug=train_aug,
            epochs=epochs,
            lr=lr,
            bsz_train=bsz_train,
            bsz_eval=bsz_eval,
            max_len=max_len,
            rdrop_alpha=rdrop_alpha,
            label_smoothing=label_smoothing,
            target_recall_spam=target_recall_spam,
            mine_topk=mine_topk,
        )

        # record current result
        state["history"].append({
            "iter": it,
            "val_f1": val_f1,
            "counts": counts,
        })
        _ensure_parent(summary_path).write_text(json.dumps(state, indent=2), encoding="utf-8")

        # early stop logic (per run)
        improved = False
        if state["global_best_f1_this_run"] is None or val_f1 > state["global_best_f1_this_run"]:
            state["global_best_f1_this_run"] = val_f1
            improved = True
            state["no_improve"] = 0
        else:
            state["no_improve"] += 1

        # 2) Prepare hard pool → generator
        hard_pool = _processed_dir() / "hard_examples.csv"
        use_hard = hard_pool.exists() and not _read_csv_safe(hard_pool, ["message", "label"]).empty

        if not use_hard:
            print("ℹ️ No hard pool available at all for this iter; generator will run in 'full' mode.")

        gen_file = run_generator(
            iter_idx=it,
            num_msgs=per_iter_new,
            mode="hard" if use_hard else "full",
            hard_file=hard_pool if use_hard else None,
        )

        # 3) Merge new augmented into cumulative train_aug
        train_aug = merge_augmented_into_train(gen_file, train_aug)

        # update summary again
        _ensure_parent(summary_path).write_text(json.dumps(state, indent=2), encoding="utf-8")

        # Stop?
        if not no_early_stop and it >= min_iters and state["no_improve"] >= patience:
            print(f"\n🛑 Early stopping: no improvement for {patience} iteration(s).")
            break

    print("\n✅ Pipeline finished.")
    print(f"Best val F1 this run: {state['global_best_f1_this_run']}")
    print(f"Summary written to:   {summary_path}")


# ======================================================================================
# CLI
# ======================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--per_iter_new", type=int, default=192)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_iters", type=int, default=1)
    parser.add_argument("--no_early_stop", action="store_true")

    parser.add_argument("--backbone", type=str, default=str(_repo_root() / "models" / "deberta_v3-base"))
    parser.add_argument("--train_base", type=str, default=str(_processed_dir() / "train_sms_dedup.csv"))
    parser.add_argument("--train_aug", type=str, default=str(_processed_dir() / "train_sms_mistral_augmented.csv"))

    parser.add_argument("--reset_summary", action="store_true")
    parser.add_argument("--exp_id", type=str, default=None)

    # Passthrough training knobs to run_discriminator
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--bsz_train", type=int, default=None)
    parser.add_argument("--bsz_eval", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--rdrop_alpha", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--target_recall_spam", type=float, default=None)
    parser.add_argument("--mine_topk", type=int, default=None)

    args = parser.parse_args()

    run_pipeline(
        iters=args.iters,
        per_iter_new=args.per_iter_new,
        patience=args.patience,
        backbone=args.backbone,
        train_base=args.train_base,
        train_aug=args.train_aug,
        min_iters=args.min_iters,
        no_early_stop=args.no_early_stop,
        reset_summary=args.reset_summary,
        exp_id=args.exp_id,
        epochs=args.epochs,
        lr=args.lr,
        bsz_train=args.bsz_train,
        bsz_eval=args.bsz_eval,
        max_len=args.max_len,
        rdrop_alpha=args.rdrop_alpha,
        label_smoothing=args.label_smoothing,
        target_recall_spam=args.target_recall_spam,
        mine_topk=args.mine_topk,
    )

if __name__ == "__main__":
    main()
