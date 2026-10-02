"""Fine-tune a multilingual cross-encoder on (query record, source-1 record) pairs and score pairs.

    python ce_model.py train <name> [--fold 0|1|all] [--model hf_id] [--epochs E] [--lr LR] [--bs B]
    python ce_model.py score <name> <split>[:fold]      # split = val | test | train (train: other fold)

Input: <s> query </s></s> source-1 </s>, each record = lower-cased raw "name | address" tokenised
once by ce_data.py.  Output: one logit per pair (BCE on the ground-truth label).
Training pairs = stage-1 filtered candidates of the training states (hard negatives come for free:
the other blocking candidates of the same query and the sibling distractors).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from common import WORK

TOK = {}


def load_tok(split, tag):
    key = (split, tag)
    if key not in TOK:
        z = np.load(os.path.join(WORK, f"tok_{split}_{tag}.npz"))
        TOK[key] = (torch.from_numpy(z["ids"].astype(np.int32)), z["off"])
    return TOK[key]


class Collator:
    """Assemble padded pair batches from the CSR token store (vectorised numpy)."""

    def __init__(self, ids, off, cls_id, sep_id, pad_id, n_sep_mid=2):
        self.ids, self.off = ids.numpy(), off
        self.cls, self.sep, self.pad, self.mid = cls_id, sep_id, pad_id, n_sep_mid

    def lengths(self, qr, sr):
        return (self.off[qr + 1] - self.off[qr]) + (self.off[sr + 1] - self.off[sr]) + 2 + self.mid

    def __call__(self, qr, sr):
        off, ids = self.off, self.ids
        ql = (off[qr + 1] - off[qr]).astype(np.int64)
        sl = (off[sr + 1] - off[sr]).astype(np.int64)
        B = len(qr)
        L = 1 + ql + self.mid + sl + 1
        mat = np.full((B, int(L.max())), self.pad, np.int64)
        mat[:, 0] = self.cls
        r = np.arange(B)

        def put(lens, starts, col0):
            n = int(lens.sum())
            if n == 0:
                return
            rows = np.repeat(r, lens)
            within = np.arange(n) - np.repeat(np.cumsum(lens) - lens, lens)
            mat[rows, np.repeat(col0, lens) + within] = ids[np.repeat(starts, lens) + within]

        put(ql, off[qr], np.ones(B, np.int64))
        for k in range(self.mid):
            mat[r, 1 + ql + k] = self.sep
        put(sl, off[sr], 1 + ql + self.mid)
        mat[r, 1 + ql + self.mid + sl] = self.sep
        att = np.arange(mat.shape[1])[None, :] < L[:, None]
        return torch.from_numpy(mat), torch.from_numpy(att.astype(np.int64))


def special_ids(model_id, text="raw"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mid = 2 if tok.__class__.__name__.startswith("XLMRoberta") else 1
    return tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id, \
        tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id, tok.pad_token_id, mid, \
        f"v{tok.vocab_size}" + ("" if text == "raw" else f"_{text}")


def build_model(model_id):
    from transformers import AutoModelForSequenceClassification
    kw = {}
    try:
        return AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1,
                                                                  attn_implementation="sdpa", **kw)
    except (ValueError, TypeError):
        return AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1, **kw)


def length_batches(lens, bs, rng, pool=64):
    """Shuffle, then sort by length inside pools of `pool` batches (less padding), shuffle batches."""
    idx = rng.permutation(len(lens))
    out = []
    step = bs * pool
    for a in range(0, len(idx), step):
        chunk = idx[a:a + step]
        chunk = chunk[np.argsort(lens[chunk], kind="stable")]
        out += [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
    order = rng.permutation(len(out))
    return [out[i] for i in order]


@torch.inference_mode()
def predict(model, col, qr, sr, bs=2048, dev="cuda"):
    model.eval()
    lens = col.lengths(qr, sr)
    order = np.argsort(lens, kind="stable")
    out = np.empty(len(qr), np.float32)
    for a in range(0, len(order), bs):
        sel = order[a:a + bs]
        x, m = col(qr[sel], sr[sel])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(input_ids=x.to(dev, non_blocking=True), attention_mask=m.to(dev, non_blocking=True)).logits
        out[sel] = lo.float().squeeze(-1).cpu().numpy()
    return out


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if 0 < y.mean() < 1 else float("nan")


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cls_id, sep_id, pad_id, mid, tag = special_ids(args.model, args.text)
    ids, off = load_tok("train", tag)
    col = Collator(ids, off, cls_id, sep_id, pad_id, mid)
    pr = pd.read_parquet(os.path.join(WORK, "pairs_train.parquet"), columns=["q_row", "s_row", "label", "fold", "qid"])
    if args.fold != "all":
        pr = pr[pr.fold == int(args.fold)].reset_index(drop=True)
    if args.max_pairs and len(pr) > args.max_pairs:
        # keep whole queries (a query's candidates are the hard negatives of its true pair)
        uq = pr.qid.unique()
        keepq = rng.choice(uq, int(len(uq) * args.max_pairs / len(pr)), replace=False)
        pr = pr[pr.qid.isin(keepq)].reset_index(drop=True)
    qr, sr, y = pr.q_row.values, pr.s_row.values, pr.label.values.astype(np.float32)
    lens = col.lengths(qr, sr)
    # monitoring subset of validation pairs (validation states: never trained on)
    va = pd.read_parquet(os.path.join(WORK, "pairs_val.parquet"), columns=["q_row", "s_row", "label", "p1"])
    va = va.sample(min(len(va), 60_000), random_state=0)
    vcol = col  # validation records live in the train split's token store

    dev = "cuda"
    model = build_model(args.model).to(dev)
    no_decay = ["bias", "LayerNorm.weight", "layernorm", "norm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, fused=True)
    steps_per_epoch = math.ceil(len(qr) / args.bs)
    total = int(steps_per_epoch * args.epochs)
    warm = max(1, int(total * 0.04))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    lossf = torch.nn.BCEWithLogitsLoss()
    out_dir = os.path.join(WORK, f"ce_{args.name}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{args.name}] model {args.model} fold {args.fold}: {len(qr):,} pairs (pos {y.mean():.3f}), "
          f"{total:,} steps of {args.bs}, lr {args.lr}", flush=True)
    step, t0, run_loss = 0, time.time(), 0.0
    ep = 0
    while step < total:
        for b in length_batches(lens, args.bs, rng):
            if step >= total:
                break
            model.train()
            x, m = col(qr[b], sr[b])
            yt = torch.from_numpy(y[b]).to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(input_ids=x.to(dev, non_blocking=True), attention_mask=m.to(dev, non_blocking=True)).logits
            loss = lossf(lo.float().squeeze(-1), yt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss = 0.98 * run_loss + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 500 == 0:
                el = time.time() - t0
                print(f"  step {step:,}/{total:,} ep {ep} loss {run_loss:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{step * args.bs / el:.0f} pairs/s", flush=True)
            if step % args.eval_every == 0 or step == total:
                s = predict(model, vcol, va.q_row.values, va.s_row.values)
                yv = va.label.values
                ll = float(np.mean(np.logaddexp(0, s) - yv * s))
                hard = (va.p1.values > 0.02) & (va.p1.values < 0.98)
                print(f"  [eval step {step:,}] val logloss {ll:.4f} auc {auc(yv, s):.5f} | hard pairs "
                      f"(0.02<p1<0.98, n={hard.sum():,}) auc {auc(yv[hard], s[hard]):.4f} "
                      f"(p1 auc {auc(yv[hard], va.p1.values[hard]):.4f})", flush=True)
        ep += 1
    model.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "ce_meta.json"), "w") as fh:
        json.dump({"model": args.model, "fold": args.fold, "pairs": int(len(qr)), "steps": total,
                   "bs": args.bs, "lr": args.lr, "epochs": args.epochs, "tag": tag, "text": args.text}, fh, indent=1)
    print(f"[{args.name}] saved {out_dir} ({time.time() - t0:.0f}s)", flush=True)


def score(args):
    out_dir = os.path.join(WORK, f"ce_{args.name}")
    meta = json.load(open(os.path.join(out_dir, "ce_meta.json")))
    split, _, fold = args.split.partition(":")
    cls_id, sep_id, pad_id, mid, tag = special_ids(meta["model"], meta.get("text", "raw"))
    tok_split = "test" if split == "test" else "train"
    ids, off = load_tok(tok_split, tag)
    col = Collator(ids, off, cls_id, sep_id, pad_id, mid)
    pr = pd.read_parquet(os.path.join(WORK, f"pairs_{split}.parquet"), columns=["q_row", "s_row", "p1"] +
                         (["fold"] if split == "train" else []))
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(out_dir, attn_implementation="sdpa").to("cuda")
    t0 = time.time()
    s = np.full(len(pr), np.nan, np.float32)
    sel = np.ones(len(pr), bool)
    if split == "train":  # out-of-fold only: rows of the fold the model was not trained on
        sel = pr.fold.values != int(meta["fold"]) if meta["fold"] != "all" else sel
    elif args.min_p1 > 0:  # only the final candidate set needs a score (NaN elsewhere)
        sel = pr.p1.values >= args.min_p1
    idx = np.nonzero(sel)[0]
    chunk = 2_000_000
    for a in range(0, len(idx), chunk):
        ii = idx[a:a + chunk]
        s[ii] = predict(model, col, pr.q_row.values[ii], pr.s_row.values[ii], bs=args.bs)
        print(f"  scored {a + len(ii):,}/{len(idx):,} ({time.time() - t0:.0f}s)", flush=True)
    np.save(os.path.join(WORK, f"ce_{args.name}_{split}.npy"), s)
    print(f"[{args.name}] {split}: {len(idx):,} pairs scored in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("name")
    ap.add_argument("split", nargs="?")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max_pairs", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text", default="raw", help="raw | tr (native scripts transliterated)")
    ap.add_argument("--min_p1", type=float, default=0.0, help="score: only val/test pairs with p1 >= this")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    else:
        a.bs = a.bs if a.bs > 256 else 2048
        score(a)
