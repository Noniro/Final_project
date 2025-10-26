import os, argparse, pandas as pd, random, re, math, sys
from pathlib import Path
from transformers import pipeline
import torch

# ── Paths (robust, relative to this file) ─────────────────────
FILE_DIR = Path(__file__).resolve().parent           # .../src/model
PROJ_ROOT = FILE_DIR.parents[1]                      # .../
DATA_DIR = PROJ_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_OUTPUT = str(DATA_DIR / "augmented_spam_v2.csv")
TRAIN_FILE = str(PROCESSED_DIR / "train_sms.csv")

HARD_FILES_DEFAULT = [
    "hard_spam_false_negatives.csv",
    "hard_spam_borderline.csv",
    "hard_spam_bottompct.csv",
]

# ── Prompts (concise, low-meta, few-shot) ─────────────────────
# We avoid words like "Output:", "Return:", "Instruction:", etc. to reduce echo.
FEWSHOT_SMS = [
    "WIN a free tablet today! Reply YES by 6pm to claim. T&Cs apply.",
    "Limited offer: 50% off your next order. Click http://deal-now.co to activate.",
    "Urgent: Your parcel fee is pending. Pay now to avoid return: www.track-pay.co/1827",
]

FULL_PROMPT = (
    "Write one short, realistic spam SMS in casual tone.\n"
    f"Examples:\n- {FEWSHOT_SMS[0]}\n- {FEWSHOT_SMS[1]}\n- {FEWSHOT_SMS[2]}\n"
    "One new SMS only. No quotes. No headings. Start directly with the text."
)

HARD_PROMPT_TEMPLATE = (
    "Write one short, realistic spam SMS similar to these:\n{examples}\n"
    "One new SMS only. No quotes. No headings. Start directly with the text."
)

# ── Regexes & filters ─────────────────────────────────────────
WRAPPER_PATTERNS = [
    r"^(?:your\s+(?:final\s+)?message|final\s+message)\s*[:\-]\s*",
    r"^(?:here(?:'|’)?s|here\s+is)\s+(?:a\s+)?(?:possible|potential|sample|example)\s*(?:spam|spammish|message|sms)?\s*[:\-]?\s*",
    r"^(?:answer|output|result)\s*[:\-]\s*",
    r"^(?:your\s+sms)\s*[:\-]\s*",
    r"^\s*sms\s*[:\-]\s*",
]

INSTRUCTION_PATTERNS = [
    r"^your\s+sms\s+should\b",
    r"^(?:the|this)\s+(?:text|message)\s+(?:should|must|needs?)\b",
    r"^(?:please\s+)?(?:generate|write|create)\b.*",
    r"^constraints?\s*[:\-]",
    r"^note\s*[:\-]",
]

BAD_CONTENT_SNIPPETS = [
    # meta/instructional
    "spammish", "sample message", "example", "examples:", "instruction",
    "return only", "output:", "you are an expert", "here's my attempt",
    "the text should", "the message should", "begin directly",
    # general non-SMS noise
    "do not include", "as an ai", "i cannot", "i'm just an ai",
]

BULLET_PAT = re.compile(r"^\s*(?:[-*•]\s+|\d+\)\s+|\(\d+\)\s+|•\s+)")
QUOTE_TRIM = re.compile(r"^[\"'`“”‘’]+|[\"'`“”‘’]+$")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}")

SPAM_KEYWORDS = [
    "win","won","prize","claim","free","call","text","reply","click",
    "http","www.","com","£","$","€","lottery","urgent","limited",
    "offer","promo","deal","voucher","gift","reward","today","now"
]

DEFAULT_HARD_PROMPTS = 12      # cap how many unique prompts we build
DEFAULT_HARD_FEW = 3          # how many examples per prompt (keep small)
EXAMPLE_TRIM_WORDS = 18        # compress examples to keep prompts short

def _trim_words(text: str, n: int) -> str:
    w = re.findall(r"\S+", str(text).strip())
    if len(w) <= n:
        return " ".join(w)
    return " ".join(w[:n]) + "…"

def _clean_line(s: str) -> str:
    s = s.replace("\r", "\n").strip()
    s = s.split("\n", 1)[0].strip()
    s = QUOTE_TRIM.sub("", s).strip()
    s = BULLET_PAT.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _strip_wrappers(s: str) -> str:
    t = s
    for pat in WRAPPER_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()
    for pat in INSTRUCTION_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^\s*(?:message|sms)\s*[:\-]\s*", "", t, flags=re.IGNORECASE).strip()
    t = QUOTE_TRIM.sub("", t).strip()
    return t

def _normalize_for_dedup(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" '\"`“”‘’")
    return s

def _looks_instructional(line: str) -> bool:
    low = line.lower().strip()
    if len(low) < 3:
        return True
    if any(re.search(p, low) for p in INSTRUCTION_PATTERNS):
        return True
    # heuristics: phrases that suggest meta/guidance
    if any(x in low for x in ["please", "constraints", "return", "do not", "output", "instruction"]):
        if len(low.split()) > 6:
            return True
    return False

def _spam_score(line: str) -> float:
    if not line or len(line) < 4:
        return -1e9
    s = line.lower()
    score = 0.0
    for kw in SPAM_KEYWORDS:
        if kw in s:
            score += 1.0
    if PHONE_RE.search(s): score += 1.5
    if "http" in s or "www." in s: score += 1.5
    w = len(s.split())
    if 3 <= w <= 24: score += 0.75
    return score

def _extract_candidate(full_text: str) -> str:
    """
    Pick one single-line SMS from arbitrary model output:
      1) split lines + quoted spans
      2) strip wrappers/quotes/bullets
      3) drop instructional lines
      4) choose highest spam_score
    """
    text = (full_text or "").replace("\r", "\n")
    candidates = []

    # primary: split lines
    for ln in text.split("\n"):
        ln = _clean_line(ln)
        if not ln:
            continue
        ln = _strip_wrappers(ln)
        ln = _clean_line(ln)
        if ln:
            candidates.append(ln)

    # secondary: quoted substrings
    quoted = re.findall(r"[\"'“‘]([^\"'”’]+)[\"'”’]", text)
    for q in quoted:
        q = _clean_line(q)
        q = _strip_wrappers(q)
        q = _clean_line(q)
        if q:
            candidates.append(q)

    filtered = [c for c in candidates if not _looks_instructional(c)]
    if not filtered:
        filtered = candidates
    if not filtered:
        return ""

    best = max(filtered, key=_spam_score)
    best = best.split("\n", 1)[0].strip()
    best = re.sub(r"\s*[-–—]\s*please.*$", "", best, flags=re.IGNORECASE).strip()
    best = _strip_wrappers(best)
    best = _clean_line(best)
    best = best.replace("\n", " ").strip()
    best = QUOTE_TRIM.sub("", best).strip()
    return best

# Softer acceptance: no hard "must contain X" list. Use score + basic sanity.
MIN_SPAM_SCORE = 1.5

def _valid_spam(s: str) -> bool:
    s = s.strip()
    if len(s) < 8 or len(s.split()) < 2:
        return False
    low = s.lower()

    if any(b in low for b in BAD_CONTENT_SNIPPETS):
        return False
    if any(re.search(p, s, flags=re.IGNORECASE) for p in WRAPPER_PATTERNS + INSTRUCTION_PATTERNS):
        return False
    if re.search(r"^\s*(?:message|sms)\s*[:\-]\s*", s, flags=re.IGNORECASE):
        return False

    return _spam_score(s) >= MIN_SPAM_SCORE

# ── Data loading helpers ───────────────────────────────────────
def load_hard_examples_from_processed(hard_files=None, min_len=6, drop_dups=True):
    if hard_files is None:
        hard_files = HARD_FILES_DEFAULT
    dfs = []
    for raw in hard_files:
        p = Path(raw.strip())
        if not p.is_absolute() and not p.exists():
            p = PROCESSED_DIR / p.name
        if not p.exists():
            print(f"ℹ️ missing: {p}")
            continue
        df = pd.read_csv(p)
        if "message" not in df.columns:
            print(f"⚠️ no 'message' column in {p} – skipping.")
            continue
        dfs.append(df[["message"]].copy())

    if not dfs:
        print("❌ No hard files found under processed/")
        return []

    hard = pd.concat(dfs, ignore_index=True)
    hard["message"] = hard["message"].astype(str).str.strip()
    hard = hard[hard["message"].str.len() >= min_len]

    if drop_dups:
        hard["_k"] = hard["message"].map(_normalize_for_dedup)
        hard = hard.drop_duplicates(subset=["_k"]).drop(columns=["_k"])

    msgs = hard["message"].tolist()
    print(f"✅ Hard pool loaded: {len(msgs)} rows (after clean/dedup) from {len(dfs)} file(s)")
    return msgs

def sample_real_spam(n=3):
    if not os.path.exists(TRAIN_FILE):
        return []
    df = pd.read_csv(TRAIN_FILE)
    df = df[df.get("label", 1) == 1]
    if df.empty:
        return []
    return random.sample(df["message"].astype(str).tolist(), min(n, len(df)))

def load_train_norm_set():
    if not os.path.exists(TRAIN_FILE):
        return set()
    df = pd.read_csv(TRAIN_FILE)
    if "message" not in df.columns:
        return set()
    msgs = df["message"].astype(str).tolist()
    return {_normalize_for_dedup(m) for m in msgs}

# ── Core generator ─────────────────────────────────────────────
def generate_spam_messages(
    generator,
    num_messages=500,
    mode="full",
    hard_examples=None,
    per_prompt=3,
    batch_size=24,
    dedup_against_train=False,
    hard_prompts=DEFAULT_HARD_PROMPTS,
    hard_few=DEFAULT_HARD_FEW,
    max_new_tokens=36,
):
    reasons = {"invalid": 0, "duplicate_in_batch": 0, "duplicate_in_train": 0}
    accepted, seen = [], set()
    train_norm_set = load_train_norm_set() if dedup_against_train else set()

    prompts = []
    if mode == "full":
        base = FULL_PROMPT
        prompts = [base] * max(1, math.ceil(num_messages / per_prompt))

    elif mode == "hard":
        if not hard_examples:
            raise ValueError("Hard mode requires a list of hard examples.")
        pool = hard_examples[:]
        random.shuffle(pool)

        # TRIM & SAMPLE: keep prompts short and few
        few_lists = []
        i = 0
        while i < len(pool) and len(few_lists) < max(1, hard_prompts):
            few = []
            for _ in range(max(1, hard_few)):
                if i >= len(pool): break
                few.append(_trim_words(pool[i], EXAMPLE_TRIM_WORDS))
                i += 1
            if few:
                few_lists.append(few)

        for few in few_lists:
            examples_str = "\n".join(f"- {m}" for m in few)
            prompts.append(HARD_PROMPT_TEMPLATE.format(examples=examples_str))

        # Fallback in case hard_prompts=0 or nothing built
        if not prompts:
            few = [_trim_words(m, EXAMPLE_TRIM_WORDS) for m in
                   random.sample(hard_examples, min(hard_few, len(hard_examples)))]
            prompts = [HARD_PROMPT_TEMPLATE.format(examples="\n".join(f"- {m}" for m in few))]

    # Pad token safety
    generator.tokenizer.pad_token = getattr(generator.tokenizer, "eos_token", generator.tokenizer.pad_token)

    # Soft bad words
    bad_words = [
        "Your message", "your message:", "Here is", "Here's",
        "Your SMS", "your sms:", "Your SMS:", "SMS:",
        "spammish", "The text should", "The message should",
    ]
    bad_words_ids = generator.tokenizer(bad_words, add_special_tokens=False).input_ids

    target = num_messages
    max_attempts = max(4, math.ceil(target * 2.0))
    attempts, idx = 0, 0

    while len(accepted) < target and attempts < max_attempts:
        if idx >= len(prompts):
            idx = 0
        batch_prompts = prompts[idx: min(idx + batch_size, len(prompts))]
        idx += batch_size
        attempts += 1

        outputs = generator(
            batch_prompts,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            top_k=40,
            repetition_penalty=1.03,
            num_return_sequences=per_prompt,
            pad_token_id=generator.tokenizer.eos_token_id if hasattr(generator.tokenizer, "eos_token_id") else None,
            bad_words_ids=bad_words_ids,
            return_full_text=False,
        )

        flat = []
        if isinstance(outputs, list):
            for item in outputs:
                flat.extend(item if isinstance(item, list) else [item])

        for d in flat:
            raw = d.get("generated_text", d.get("text", ""))
            candidate = _extract_candidate(raw)
            if not candidate:
                reasons["invalid"] += 1
                continue

            candidate = _clean_line(_strip_wrappers(_clean_line(candidate)))

            if not _valid_spam(candidate):
                reasons["invalid"] += 1
                continue

            norm = _normalize_for_dedup(candidate)
            if norm in seen:
                reasons["duplicate_in_batch"] += 1
                continue
            if norm in train_norm_set:
                reasons["duplicate_in_train"] += 1
                continue

            seen.add(norm)
            accepted.append(candidate)
            if len(accepted) >= target:
                break

    print(f"kept {len(accepted)}/{target} | dropped: {reasons}")
    return list(dict.fromkeys(accepted))

# ── CLI ───────────────────────────────────────────────────────
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="full", choices=["full", "hard"])
    ap.add_argument("--num_messages", type=int, default=None)
    ap.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    ap.add_argument("--hard_files", type=str, default="")
    ap.add_argument("--per_prompt", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--dedup_train", action="store_true")
    ap.add_argument("--hard_prompts", type=int, default=DEFAULT_HARD_PROMPTS,
                    help="Max number of unique prompts to build in hard mode.")
    ap.add_argument("--hard_few", type=int, default=DEFAULT_HARD_FEW,
                    help="Number of examples per hard prompt (keep small for speed).")
    ap.add_argument("--max_new_tokens", type=int, default=36,
                    help="Max new tokens per SMS generation.")
    return ap
def main():
    args = build_argparser().parse_args()
    if args.num_messages is None:
        args.num_messages = 2000 if args.mode == "full" else 500

    hard_examples = None
    if args.mode == "hard":
        files_list = [s.strip() for s in args.hard_files.split(",") if s.strip()] if args.hard_files.strip() else None
        hard_examples = load_hard_examples_from_processed(files_list)
        print("Incoming hard examples actually used:", len(hard_examples))
        if not hard_examples:
            pd.DataFrame(columns=["message", "label"]).to_csv(args.output, index=False)
            print(f"✅ Generated 0 messages in hard mode → saved to {args.output}")
            return

    # Use dtype (not torch_dtype) to avoid the deprecation warning
    gen = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        device=0,
        return_full_text=False,
        dtype=torch.float16,
    )
    gen.tokenizer.pad_token = getattr(gen.tokenizer, "eos_token", gen.tokenizer.pad_token)

    messages = generate_spam_messages(
        gen,
        num_messages=args.num_messages,
        mode=args.mode,
        hard_examples=hard_examples,
        per_prompt=args.per_prompt,
        batch_size=args.batch_size,
        dedup_against_train=args.dedup_train,
        hard_prompts=args.hard_prompts,
        hard_few=args.hard_few,
        max_new_tokens=args.max_new_tokens,
    )

    data_to_save = [{'message': msg, 'label': 1} for msg in messages]
    df_out = pd.DataFrame(data_to_save)
    print(f"✅ Generated {len(messages)} messages in {args.mode} mode → saved to {args.output}")

if __name__ == "__main__":
    main()
