#!/usr/bin/env python3
"""Formal paired bootstrap difference-of-differences from canonical predictions."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from advanced_common import LABELS, RANDOM_STATE


def load(path):
    d = pd.read_csv(path)
    return d.id.astype(str).to_numpy(), d.true_label.to_numpy(int), d.predicted_label.to_numpy(int)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--predictions-dir", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--bootstrap", type=int, default=5000); args=ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "S75": args.predictions_dir / "pred_Statistical_28_h75.csv",
        "T75": args.predictions_dir / "pred_Full_temporal_82_h75.csv",
        "S100": args.predictions_dir / "pred_Statistical_28_h100.csv",
        "T100": args.predictions_dir / "pred_Full_temporal_82_h100.csv",
    }
    loaded={k:load(v) for k,v in files.items()}; ids,y,_=loaded["S75"]
    for k,(ids2,y2,_) in loaded.items():
        if not np.array_equal(ids,ids2) or not np.array_equal(y,y2): raise RuntimeError(f"Pairing mismatch: {k}")
    pred={k:v[2] for k,v in loaded.items()}
    fn=lambda yt,yp:f1_score(yt,yp,labels=LABELS,average="macro",zero_division=0)
    s={k:fn(y,p) for k,p in pred.items()}
    d75=s["T75"]-s["S75"]; d100=s["T100"]-s["S100"]; interaction=d75-d100
    rng=np.random.default_rng(RANDOM_STATE); vals=np.empty(args.bootstrap)
    for b in range(args.bootstrap):
        idx=rng.integers(0,len(y),len(y)); yy=y[idx]
        dd75=fn(yy,pred["T75"][idx])-fn(yy,pred["S75"][idx])
        dd100=fn(yy,pred["T100"][idx])-fn(yy,pred["S100"][idx])
        vals[b]=dd75-dd100
    out={"statistical_75_macro_f1":s["S75"],"temporal_75_macro_f1":s["T75"],"effect_75_temporal_minus_statistical":d75,
         "statistical_100_macro_f1":s["S100"],"temporal_100_macro_f1":s["T100"],"effect_100_temporal_minus_statistical":d100,
         "interaction_difference_of_differences":interaction,"ci95_low":float(np.quantile(vals,.025)),"ci95_high":float(np.quantile(vals,.975)),"bootstrap_samples":args.bootstrap,
         "definition":"(Temporal75-Statistical75) - (Temporal100-Statistical100)"}
    (args.output_dir/"formal_interaction.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    pd.DataFrame({"bootstrap_interaction":vals}).to_csv(args.output_dir/"formal_interaction_bootstrap.csv",index=False)
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
