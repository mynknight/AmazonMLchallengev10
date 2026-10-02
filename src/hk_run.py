"""US/India final model with V9's hard-pair cross-encoders (e5hk) instead of e5k4.

    python hk_run.py ea        # WORK10/ea_hk/ce_{train,val,test}.npy = mean of ea/ce_hk_f0..3 (NaN = not out-of-fold)
    python hk_run.py train     # train9 k4hk_u_ea (as k4_u_ea, --ce e5hk_f0..3)  -> V9 work/k4hk_u_ea*
    python hk_run.py predict   # predict9 US/India test -> WORK10/test_usin_k4hk_u_ea.parquet

V9's train9/predict9 read the empty-address rows' CE from the fixed path WORK9/ea/ce_<split>.npy;
this wrapper redirects exactly those files to the hk versions, so V9's code and files are unchanged
(new outputs only get the new name k4hk_u_ea).
"""
import argparse
import os
import sys

import numpy as np

from common10 import CE4, WORK10, WORK9

EA_HK = os.path.join(WORK10, "ea_hk")
NAME, CE = "k4hk_u_ea", "e5hk_f0+e5hk_f1+e5hk_f2+e5hk_f3"


def build_ea():
    os.makedirs(EA_HK, exist_ok=True)
    for split in ("train", "val", "test"):
        with np.errstate(all="ignore"):
            s = np.nanmean([np.load(os.path.join(WORK9, "ea", f"ce_hk_f{f}_{split}.npy")) for f in range(4)], axis=0)
        assert not np.isnan(s).any(), split
        np.save(os.path.join(EA_HK, f"ce_{split}.npy"), s.astype(np.float32))
        print(f"ea hk {split}: {len(s):,} rows", flush=True)


def redirect_ea():
    real = np.load
    ea = os.path.join(WORK9, "ea")

    def load(path, *args, **kw):
        p = os.fspath(path)
        if os.path.dirname(p) == ea and os.path.basename(p) in ("ce_train.npy", "ce_val.npy", "ce_test.npy"):
            p = os.path.join(EA_HK, os.path.basename(p))
            print(f"  (ea CE -> {p})", flush=True)
        return real(p, *args, **kw)
    np.load = load


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "ea":
        build_ea()
    elif step == "train":
        redirect_ea()
        import train9
        train9.main(argparse.Namespace(name=NAME, ce=CE, no_w=True, u=True, self_train=False, ea=True, seed=0, rounds=4000))
    elif step == "predict":
        redirect_ea()
        import predict9
        predict9.main(argparse.Namespace(model=NAME, out=os.path.join(WORK10, f"test_usin_{NAME}.parquet"),
                                         ce=CE, ce_dir=CE4, countries="US,India", ea=True))
