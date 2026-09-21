from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def fmt(x, n=4):
    try: return f"{float(x):.{n}f}"
    except Exception: return str(x)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results-dir',type=Path,required=True); args=ap.parse_args()
    r=args.results_dir; lines=["# Advanced DGA strengthening run summary",""]
    env=r/'adaptive'/'environment.json'
    if env.exists():
        e=json.loads(env.read_text()); lines += [f"Environment: Python `{e.get('python','').split()[0]}`, scikit-learn `{e.get('scikit_learn')}`, NumPy `{e.get('numpy')}`, pandas `{e.get('pandas')}`, SHAP `{e.get('shap')}`.",""]
    p=r/'interaction'/'formal_interaction.json'
    if p.exists():
        d=json.loads(p.read_text()); lines += ["## Formal interaction",f"Difference-of-differences = **{fmt(d['interaction_difference_of_differences'])}**, paired 95% CI **[{fmt(d['ci95_low'])}, {fmt(d['ci95_high'])}]**.",""]
    p=r/'adaptive'/'adaptive_policy_summary.json'
    if p.exists():
        d=json.loads(p.read_text()); t=d['test_result']; lines += ["## Adaptive stopping",f"Training-OOF selected threshold = **{d['selected_threshold']}**, persistence = **{d['selected_persistence']}** horizons. Test macro-F1 = **{fmt(t['macro_f1'])}**, endpoint retention = **{100*t['retention_vs_endpoint']:.2f}%**, mean stop = **{t['mean_stop_percent']:.1f}%**, median stop = **{t['median_stop_percent']:.1f}%**, stopped before 100% = **{100*t['fraction_stopped_before_100']:.1f}%**.",""]
    p=r/'dense_representation'/'dense_representation_curve.csv'
    if p.exists():
        d=pd.read_csv(p); x=d[d.representation=='Temporal_minus_Statistical']; best=x.loc[x.test_macro_f1.idxmax()]; lines += ["## Dense representation curve",f"Largest observed temporal-minus-statistical test macro-F1 = **{fmt(best.test_macro_f1)}** at **{int(best.horizon_percent)}%** history; paired CI **[{fmt(best.paired_ci95_low)}, {fmt(best.paired_ci95_high)}]**.",""]
    p=r/'ablation'/'feature_family_ablation_75_summary.csv'
    if p.exists():
        d=pd.read_csv(p).sort_values('gain_when_added_to_stat',ascending=False); b=d.iloc[0]; lines += ["## Feature-family ablation at 75%",f"Largest gain when added to Statistical-28: **{b['group']}**, Δ macro-F1 **{fmt(b['gain_when_added_to_stat'])}**, CI **[{fmt(b['gain_ci95_low'])}, {fmt(b['gain_ci95_high'])}]**.",""]
    p=r/'robustness'/'robustness_summary.csv'
    if not p.exists(): p=r/'robustness20'/'robustness_summary.csv'
    if p.exists():
        d=pd.read_csv(p); lines += ["## Robustness stress test"]
        for _,z in d.iterrows(): lines.append(f"- {z.scenario}, severity {z.severity:g}: macro-F1 {fmt(z.macro_f1_mean)} ± {fmt(z.macro_f1_std)}, Δ vs clean {fmt(z.macro_f1_change_mean)}.")
        lines.append("")
    p=r/'conformal'/'conformal_by_horizon.csv'
    if p.exists():
        d=pd.read_csv(p); lines += ["## Fixed-horizon class-conditional conformal diagnostics"]
        for h in [50,75,100]:
            z=d[d.horizon_percent==h].iloc[0]; lines.append(f"- {h}%: coverage **{100*z.marginal_coverage:.2f}%**, singleton rate **{100*z.singleton_rate:.2f}%**, mean set size **{z.mean_set_size:.3f}**.")
        lines.append("")
    p=r/'shap'/'shap_stability.csv'
    if p.exists():
        d=pd.read_csv(p); lines += ["## SHAP stability"]
        for _,z in d.iterrows(): lines.append(f"- {int(z.horizon_a)}% vs {int(z.horizon_b)}%: Spearman ρ **{z.spearman_rho_all_features:.3f}**, top-10 Jaccard **{z.top10_jaccard:.3f}**.")
        lines.append("")
    lines += ["## Interpretation warning","Only the formal interaction section can use the existing canonical `final_5fold` predictions directly. Dense-horizon, adaptive, robustness, conformal, ablation, and early-SHAP numbers are final only when the suite is run in the locked canonical environment (scikit-learn 1.9.0) and that environment is archived."]
    out=r/'ADVANCED_RESULTS_SUMMARY.md'; out.write_text('\n'.join(lines),encoding='utf-8'); print(out)
if __name__=='__main__': main()
