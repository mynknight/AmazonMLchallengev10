"""France-adapted cross-encoders for the V10 France veto (self-training on v6a's France decisions).

    python ce_fr10.py train <fold>   # fine-tune V9's ce_e5k4_f<fold> -> work/ce_fr/ce_e5fr_f<fold>
    python ce_fr10.py score <fold>   # France test pairs (p1 >= 0.01) -> work/ce_fr/ce_e5fr_f<fold>_test.npy
                                     # + US/India validation log-loss (forgetting check)

The e5k4 cross-encoders misread French words (they accept only 36% of French noise-word pairs).
Pseudo-labels come from v6a's France model (fin_nv_u, threshold 0.82), which the leaderboard
confirmed (France ~0.946 -> ~0.970): a France pair is positive when it is its query's best candidate
with prob >= POS, negative when prob <= NEG; the uncertain band is never trained on.  They are
mixed 1:1 with labelled US/India training pairs so the model keeps its US/India behaviour.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common10 import CE4, WORK10  # noqa: E402
from ce_model4 import Collator, length_batches, predict, special_ids  # noqa: E402

OUT = os.path.join(WORK10, "ce_fr")
MODEL = "intfloat/multilingual-e5-small"
POS, NEG = 0.98, 0.02
N_FR, N_USIN = 1_200_000, 1_200_000
LR, BS = 3e-5, 256


def stores():
    cls_id, sep_id, pad_id, mid, tag = special_ids(MODEL, "tr")
    a = np.load(os.path.join(CE4, f"tok_train_{tag}.npz"))
    b = np.load(os.path.join(CE4, f"tok_test_{tag}.npz"))
    n_train = len(a["off"]) - 1
    ids = np.concatenate([a["ids"], b["ids"]]).astype(np.int32)
    off = np.concatenate([a["off"], b["off"][1:] + len(a["ids"])])
    return Collator(torch.from_numpy(ids), off, cls_id, sep_id, pad_id, mid), n_train


def france_pairs():
    pr = pd.read_parquet(os.path.join(CE4, "pairs_test.parquet"), columns=["qid", "s1", "q_row", "s_row", "country", "p1"])
    fr = pr[(pr.country == "France") & (pr.p1 >= 0.01)]
    sc = pd.read_parquet(os.path.join(WORK10, "test_fr_v6a.parquet"), columns=["qid", "s1", "prob"])
    return fr.merge(sc, on=["qid", "s1"], how="inner", validate="one_to_one")


def pseudo(fr):
    fr = fr.sort_values(["qid", "prob"], ascending=[True, False])
    top = ~fr.qid.duplicated()
    pos, neg = fr[top & (fr.prob >= POS)], fr[fr.prob <= NEG]
    npos = min(len(pos), N_FR // 2); nneg = min(len(neg), N_FR - npos)
    print(f"France pseudo-labels (v6a): {npos:,} positives of {len(pos):,}, {nneg:,} negatives of {len(neg):,}", flush=True)
    return pd.concat([pos.sample(npos, random_state=0).assign(label=1), neg.sample(nneg, random_state=0).assign(label=0)])


def train(fold):
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(fold); rng = np.random.default_rng(fold)
    col, n_train = stores()
    fr = pseudo(france_pairs())
    us = pd.read_parquet(os.path.join(CE4, "pairs_train.parquet"), columns=["q_row", "s_row", "label", "fold"])
    us = us[us.fold != fold].sample(N_USIN, random_state=fold)
    qr = np.r_[us.q_row.values, fr.q_row.values + n_train]
    sr = np.r_[us.s_row.values, fr.s_row.values + n_train]
    y = np.r_[us.label.values, fr.label.values].astype(np.float32)
    lens = col.lengths(qr, sr)
    from transformers import AutoModelForSequenceClassification
    src = os.path.join(CE4, f"ce_e5k4_f{fold}")
    model = AutoModelForSequenceClassification.from_pretrained(src, attn_implementation="sdpa").to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, fused=True)
    total = int(np.ceil(len(qr) / BS)); warm = max(1, int(total * 0.04))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    lossf = torch.nn.BCEWithLogitsLoss()
    print(f"[e5fr_f{fold}] {len(us):,} US/India + {len(fr):,} France pairs, {total:,} steps", flush=True)
    step, t0, run = 0, time.time(), 0.0
    for b in length_batches(lens, BS, rng):
        model.train()
        x, m = col(qr[b], sr[b])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(input_ids=x.cuda(non_blocking=True), attention_mask=m.cuda(non_blocking=True)).logits
        loss = lossf(lo.float().squeeze(-1), torch.from_numpy(y[b]).cuda())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 1000 == 0:
            print(f"  step {step:,}/{total:,} loss {run:.4f} {step * BS / (time.time() - t0):.0f} pairs/s", flush=True)
    out = os.path.join(OUT, f"ce_e5fr_f{fold}")
    model.save_pretrained(out)
    json.dump({"init": src, "pseudo_from": "v6a fin_nv_u", "pos": POS, "neg": NEG, "lr": LR, "fold": fold,
               "model": MODEL, "text": "tr"}, open(os.path.join(out, "ce_meta.json"), "w"), indent=1)
    print(f"[e5fr_f{fold}] saved ({time.time() - t0:.0f}s)", flush=True)


def score(fold):
    from transformers import AutoModelForSequenceClassification
    col, n_train = stores()
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(OUT, f"ce_e5fr_f{fold}"),
                                                               attn_implementation="sdpa").to("cuda")
    pr = pd.read_parquet(os.path.join(CE4, "pairs_test.parquet"), columns=["q_row", "s_row", "country", "p1"])
    s = np.full(len(pr), np.nan, np.float32)
    sel = np.nonzero(((pr.country == "France") & (pr.p1 >= 0.01)).values)[0]
    s[sel] = predict(model, col, pr.q_row.values[sel] + n_train, pr.s_row.values[sel] + n_train)
    np.save(os.path.join(OUT, f"ce_e5fr_f{fold}_test.npy"), s)
    va = pd.read_parquet(os.path.join(CE4, "pairs_val.parquet"), columns=["q_row", "s_row", "label"])
    v = predict(model, col, va.q_row.values, va.s_row.values)
    old = np.load(os.path.join(CE4, f"ce_e5k4_f{fold}_val.npy"))
    ll = lambda z: float(np.mean(np.logaddexp(0, z) - va.label.values * z))
    print(f"[e5fr_f{fold}] France pairs scored {len(sel):,}; US/India val log-loss {ll(v):.4f} (e5k4 {ll(old):.4f})", flush=True)


if __name__ == "__main__":
    {"train": train, "score": score}[sys.argv[1]](int(sys.argv[2]))
