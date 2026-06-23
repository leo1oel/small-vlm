"""Master cross-model synthesis for the encoder-free vs encoder-based fusion
study. Reads every per-model result file and emits:
  - a table of functional fusion (FDI/funcCoM_rel, R0, intact, swap)
  - visual-pathway maturation summaries (image vs text update CoM, early-image
    update share, layer-1 u_img/u_txt ratio)
  - FastV attention (last->image mean/peak, sink)
and saves the cross-model figures. Robust to missing files (prints MISSING).

Run with the neo venv python (matplotlib): /envs/neo/bin/python.
Usage: python devtools/analysis_master.py
"""

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_crossmodel_figure import curves  # reuse the exact FDI/curves logic

# (name, family, N, prebuf_rel, functional_file, maturation_file)
ROSTER = [
    ("NEO1.0-2B-SFT", "free", 40, 12/40, "results_fusion_full_neo_strat.jsonl", "results_mat_neo_2B-SFT.jsonl"),
    ("NEO1.0-2B-MT",  "free", 40, 12/40, "results_fusion_full_neo_MT.jsonl", "results_mat_neo_2B-MT.jsonl"),
    ("NEO1.0-9B-SFT", "free", 42, 6/42,  "results_fusion_full_neo_NEO1_0-9B-SFT.jsonl", "results_mat_neo_9B-SFT.jsonl"),
    ("NEO1.0-9B-MT",  "free", 42, 6/42,  "results_fusion_full_neo_NEO1_0-9B-MT.jsonl", "results_mat_neo_9B-MT.jsonl"),
    ("NEO1.5-2B-SFT", "free", 40, 12/40, "results_fusion_full_neo_NEO1_5-2B-SFT.jsonl", "results_mat_neo_15-2B-SFT.jsonl"),
    ("NEO1.5-9B-SFT", "free", 42, 6/42,  "results_fusion_full_neo_NEO1_5-9B-SFT.jsonl", "results_mat_neo_15-9B-SFT.jsonl"),
    ("Gemma-4-12B",   "free", 48, None,  "results_fusion_full_gemma4_strat.jsonl", "results_mat_gemma.jsonl"),
    ("SAIL-7B",       "free", 32, None,  "results_fusion_full_sail.jsonl", "results_mat_sail.jsonl"),
    ("LLaVA-1.5-7B",  "enc",  32, None,  "results_fusion_full_llava.jsonl", "results_mat_llava.jsonl"),
    ("Qwen2.5-VL-7B", "enc",  28, None,  "results_fusion_full_qwenvl.jsonl", "results_mat_qwen.jsonl"),
]


def load(path):
    p = ROOT / path
    if not p.exists() or p.stat().st_size == 0:
        return None
    return [json.loads(l) for l in open(p) if l.strip() and '"skip"' not in l]


def mat_summary(rows):
    rows = [r for r in rows if "u_img" in r]
    if not rows:
        return None
    N = len(rows[0]["u_img"])
    rows = [r for r in rows if len(r["u_img"]) == N]
    ui = [mean(r["u_img"][l] for r in rows) for l in range(N)]
    ut = [mean(r["u_txt"][l] for r in rows) for l in range(N)]
    ul = [mean(r["u_last"][l] for r in rows) for l in range(N)]
    ni = [mean(r["norm_img"][l] for r in rows) for l in range(N)]

    def com_rel(u):
        tot = sum(u) or 1e-9
        return sum((l + 0.5) / N * u[l] for l in range(N)) / tot
    early = max(1, N // 4)
    img_early_share = sum(ui[:early]) / (sum(ui) or 1e-9)
    txt_early_share = sum(ut[:early]) / (sum(ut) or 1e-9)
    return dict(N=N, n=len(rows), u_img=ui, u_txt=ut, u_last=ul, norm_img=ni,
                img_com_rel=com_rel(ui), txt_com_rel=com_rel(ut),
                img_early_share=img_early_share, txt_early_share=txt_early_share,
                ratio_L1=ui[0] / (ut[0] or 1e-9),
                ratio_early=(sum(ui[:early]) / early) / ((sum(ut[:early]) / early) or 1e-9))


def main():
    print("=" * 110)
    print(f"{'model':16} {'fam':4} {'N':3} {'n/nc':9} {'intact':6} {'swap':6} "
          f"{'R0':6} {'funcCoM_rel(FDI)':16} {'imgCoM':7} {'imgEarly%':9} {'uimg/utxt@L1':12}")
    print("-" * 110)
    master = {}
    for name, fam, N, prebuf, ffile, mfile in ROSTER:
        frows = load(ffile)
        mrows = load(mfile)
        rec = dict(name=name, family=fam, N=N, prebuf_rel=prebuf)
        if frows and any("cost" in r for r in frows):
            c = curves(frows)
            rec["func"] = {k: c[k] for k in ("R0", "intact", "swap", "com_rel", "retained", "rho", "n", "nc", "N")}
            fcom = f"{c['com_rel']:.3f}"
            ist, sw, r0 = f"{c['intact']:.3f}", f"{c['swap']:.3f}", f"{c['R0']:+.3f}"
            ncs = f"{c['n']}/{c['nc']}"
        else:
            fcom = ist = sw = r0 = ncs = "--"
        if mrows:
            m = mat_summary(mrows)
            if m:
                rec["mat"] = m
                icom = f"{m['img_com_rel']:.3f}"
                ish = f"{m['img_early_share']:.3f}"
                rat = f"{m['ratio_L1']:.2f}"
            else:
                icom = ish = rat = "--"
        else:
            icom = ish = rat = "--"
        master[name] = rec
        print(f"{name:16} {fam:4} {N:3} {ncs:9} {ist:6} {sw:6} {r0:6} {fcom:16} {icom:7} {ish:9} {rat:12}")
    print("=" * 110)
    (ROOT / "master_results.json").write_text(json.dumps(master, indent=1))
    print("wrote master_results.json")

    # attention summary
    print("\n--- FastV attention (last->image) ---")
    for jf in ["results_attn_llava_full.json", "results_attn_qwen_full.json",
               "results_attn_gemma.json", "results_sail_attn.json",
               "results_encoder_vlm_attn.json", "results_attn_mass.json"]:
        p = ROOT / jf
        if not p.exists() or p.stat().st_size <= 2:
            continue
        d = json.loads(p.read_text())
        for k, v in d.items():
            if isinstance(v, dict) and "last2img" in v:
                img = v["last2img"]
                print(f"  {k:34} n={v.get('n')} last->img mean={mean(img):.3f} "
                      f"peak={max(img):.3f}@L{img.index(max(img))}")
            elif isinstance(v, dict) and "masses" in v:  # NEO attn_mass (POPE)
                cl = v["classes"]; li_, vi = cl.index("last"), cl.index("vision")
                masses = v["masses"]
                l2v = [masses[l][li_][vi] for l in range(len(masses))]
                print(f"  NEO-{k}(POPE) n={v.get('n')} last->vision mean={mean(l2v):.3f} "
                      f"peak={max(l2v):.3f}@L{l2v.index(max(l2v))}")


if __name__ == "__main__":
    main()
