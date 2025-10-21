# ood_utils.py
import numpy as np
import torch
from torch.nn import functional as F

@torch.no_grad()
def extract_features_and_logits(model, tokenizer, texts, max_len, device, batch_size=128):
    """Return (features, logits) for texts. Features = pooled hidden (CLS) from last layer."""
    feats, logits_all = [], []
    model.eval()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, truncation=True, padding="longest",
                        max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        # take CLS from last hidden state: [B, L, H] -> [B, H]
        last = out.hidden_states[-1][:, 0]
        feats.append(last.detach().cpu())
        logits_all.append(out.logits.detach().cpu())
        del enc, out, last
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    feats = torch.cat(feats).numpy()
    logits = torch.cat(logits_all).numpy()
    return feats, logits

def msp_from_logits(logits):
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    return 1.0 - probs.max(axis=1)  # higher = more OOD-like

def energy_from_logits(logits):
    # negative logsumexp; lower = more in-domain/confident
    x = torch.tensor(logits)
    e = -torch.logsumexp(x, dim=1)
    return e.numpy()

def fit_mahalanobis(features, labels):
    """Fit class means and a tied covariance on in-domain features."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    means = {c: features[labels == c].mean(axis=0) for c in classes}
    # tied cov
    diffs = []
    for c in classes:
        f = features[labels == c]
        diffs.append(f - means[c])
    diffs = np.vstack(diffs)
    cov = np.cov(diffs, rowvar=False) + 1e-6*np.eye(diffs.shape[1])
    VI = np.linalg.inv(cov)
    return means, VI

def mahalanobis_score(features, means, VI):
    """Minimum class-conditional Mahalanobis distance (smaller = more in-domain)."""
    dists = []
    for c, mu in means.items():
        diff = features - mu[None, :]
        d = np.einsum("bi,ij,bj->b", diff, VI, diff)
        dists.append(d)
    dmin = np.min(np.stack(dists, axis=1), axis=1)
    return dmin  # higher = more OOD-like
