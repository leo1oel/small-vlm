"""Collect every model's fusion-depth (sufmeanabl q50 + IQR + R0) across VMCBench / MMStar /
WorldBench into one clean JSON, the data backbone for the persuasive figure set.
CPU-light (jsonl means). Usage: python devtools/fig_data_collect.py
"""

import json
from pathlib import Path

from patch_analysis import load, patch_quants  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"

# tag -> (label, family, is_moe, note). family in {encoder, encoder-free, unified}.
MODELS = {
    "qwen": ("Qwen2.5-VL-7B", "encoder", False, ""),
    "ivl2": ("InternVL3.5-2B", "encoder", False, ""),
    "ivl4": ("InternVL3.5-4B", "encoder", False, ""),
    "ivl8": ("InternVL3.5-8B", "encoder", False, ""),
    "ivl30moe": ("InternVL3.5-30B-A3B", "encoder", True, ""),
    "gemma12": ("Gemma-4-12B", "encoder-free", False, "unified (no deep vision tower)"),
    "gemma26moe": ("Gemma-4-26B-MoE", "encoder", True, "ships SigLIP encoder (gemma4_vision)"),
    "neo2b": ("NEO1.0-2B", "encoder-free", False, ""),
    "neo9b": ("NEO1.0-9B", "encoder-free", False, ""),
    "neo15_2b": ("NEO1.5-2B", "encoder-free", False, ""),
    "neo15_9b": ("NEO1.5-9B", "encoder-free", False, ""),
    "mono": ("Mono-InternVL-2B", "encoder-free", False, ""),
    "sail": ("SAIL-7B", "encoder-free", False, ""),
    "janus": ("Janus-Pro-7B", "unified", False, ""),
    "llava": ("LLaVA-1.5-7B", "encoder", False, "outlier (small-scale frozen-encoder recipe)"),
}
# knockout phi q50 (prefix-knockout), for the orthogonal-metric agreement panel
PHI = {"llava": 0.33, "qwen": 0.54, "neo2b": 0.54, "sail": 0.59, "gemma12": 0.48, "ivl8": 0.44}


def suf_q(path):
    p = ROOT / path
    if not p.exists():
        return None
    pq = patch_quants(load(p))
    s = pq.get("sufmeanabl")
    if not s:
        return None
    return {
        "q50": s["q50"],
        "q25": s["q25"],
        "q75": s["q75"],
        "R0": pq.get("R0"),
        "n": pq.get("n_sufmeanabl"),
    }


def main():
    out = {"models": {}, "phi": PHI}
    for tag, (label, fam, moe, note) in MODELS.items():
        e = {"label": label, "family": fam, "moe": moe, "note": note}
        e["vmcbench"] = suf_q(f"results_sufpatch_{tag}.jsonl")
        e["mmstar"] = suf_q(f"results_mmstar_{tag}.jsonl")
        wb = ROOT / f"wb_domain_wb_{tag}.json"
        if wb.exists():
            d = json.load(open(wb))
            e["worldbench"] = {
                "q50": d["overall"]["q50"],
                "R0": d["overall"]["R0"],
                "by_domain": {
                    k: {"q50": v["q50"], "R0": v["R0"], "n": v["n"]}
                    for k, v in d["by_domain"].items()
                },
            }
        out["models"][tag] = e
        v = e["vmcbench"]
        print(
            f"{tag:11s} {label:22s} {fam:13s} VMC={v['q50'] if v else '-'} "
            f"MMStar={e['mmstar']['q50'] if e['mmstar'] else '-'} "
            f"WB={e.get('worldbench', {}).get('q50', '-')}",
            flush=True,
        )
    (ROOT / "fusion_depth_all.json").write_text(json.dumps(out, indent=2))
    print(f"saved {ROOT / 'fusion_depth_all.json'}", flush=True)


if __name__ == "__main__":
    main()
