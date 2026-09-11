#!/usr/bin/env python3
"""Fail fast if the checked-in evidence no longer matches the paper's key results."""
from pathlib import Path
import json, math, pandas as pd
R=Path(__file__).resolve().parents[1]

def close(a,b,tol=5e-7):
    if not math.isclose(float(a),float(b),rel_tol=0,abs_tol=tol):
        raise AssertionError(f"{a} != {b}")

interaction=json.load(open(R/"evidence/advanced/interaction/formal_interaction.json"))
close(interaction["interaction_difference_of_differences"],0.0760661044116)
close(interaction["ci95_low"],0.027847059177085645)
close(interaction["ci95_high"],0.12631159251265753)

adaptive=json.load(open(R/"evidence/advanced/adaptive/adaptive_policy_summary.json"))
assert adaptive["selected_threshold"]==0.95 and adaptive["selected_persistence"]==3
close(adaptive["test_result"]["macro_f1"],0.8865367051013869)
close(adaptive["test_result"]["mean_stop_percent"],37.76111111111111)

ref=json.load(open(R/"evidence/refinement/03_noise_robust_selection.json"))
assert ref["selected_causal_moving_average_window"]==11
sel=json.load(open(R/"evidence/refinement/01_selected_model.json"))
assert sel["C"]==100.0 and sel["class_weight"]=="balanced"

noise=pd.read_csv(R/"evidence/refinement/04_noise_test_summary.csv")
row=noise[(noise.severity==0.02)&(noise.model=="Temporal-70 robust w=11")].iloc[0]
close(row.macro_f1_mean,0.6250262543199961,2e-6)

print("Evidence verification PASSED")
