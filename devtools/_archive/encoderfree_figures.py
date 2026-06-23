"""Figures for the encoder-free vs encoder-based fusion study. Reads
neo_analysis/master_results.json (written by analysis_master.py) and the
attention jsons. Produces three PNGs in neo_report/.

Run with neo venv python (matplotlib). Usage: python devtools/encoderfree_figures.py
"""

import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report"
OUT.mkdir(exist_ok=True)
M = json.loads((ROOT / "master_results.json").read_text())
FREE = "#1f77b4"
ENC = "#ff7f0e"


def order(items):
    return sorted(items, key=lambda kv: (kv[1]["family"] != "free", kv[0]))


def fig_fdi():
    rows = [(k, v) for k, v in M.items() if "func" in v]
    rows = order(rows)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    # left: FDI dot chart
    ys = list(range(len(rows)))
    for y, (k, v) in zip(ys, rows):
        c = FREE if v["family"] == "free" else ENC
        ax[0].plot(v["func"]["com_rel"], y, "o", ms=12, color=c)
        ax[0].text(v["func"]["com_rel"], y + 0.18, f"{v['func']['com_rel']:.2f}",
                   ha="center", fontsize=8)
    ax[0].axvspan(0.45, 0.55, color="gray", alpha=.15)
    ax[0].axvline(0.33, color="k", ls=":", lw=.7); ax[0].axvline(0.66, color="k", ls=":", lw=.7)
    ax[0].set_yticks(ys); ax[0].set_yticklabels([k for k, _ in rows], fontsize=9)
    ax[0].set_xlim(0, 1); ax[0].set_xlabel("functional fusion CoM (relative depth) = FDI")
    ax[0].set_title("WHERE the image is USED for the answer\n(blue=encoder-free, orange=encoder; gray=mid 0.45-0.55)")
    ax[0].invert_yaxis()
    # right: retained(d)/R0 curves
    for k, v in rows:
        f = v["func"]; N = f["N"]; R0 = f["R0"] or 1e-9
        x = [(i + 1) / N for i in range(N)]
        rn = [f["retained"][i] / R0 for i in range(N)]
        c = FREE if v["family"] == "free" else ENC
        ax[1].plot(x, rn, "-", lw=1.5, color=c, alpha=.8,
                   label=f"{k} ({f['com_rel']:.2f})")
    ax[1].axhline(0.5, color="k", lw=.6, ls="--", alpha=.4)
    ax[1].axvspan(0.45, 0.55, color="gray", alpha=.12)
    ax[1].set_xlabel("relative blocked depth d/N"); ax[1].set_ylabel("retained(d)/R0")
    ax[1].set_title("usable image signal surviving text->image block [0..d)")
    ax[1].legend(fontsize=6.5, ncol=1, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "fig_fdi.png", dpi=130); print("wrote fig_fdi.png")


def fig_maturation():
    rows = [(k, v) for k, v in M.items() if "mat" in v]
    rows = order(rows)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for k, v in rows:
        m = v["mat"]; N = m["N"]
        x = [(i + 0.5) / N for i in range(N)]
        c = FREE if v["family"] == "free" else ENC
        # normalize each curve by its own mean to compare SHAPE (norm-robust)
        ui = m["u_img"]; mu = mean(ui) or 1e-9
        ax[0].plot(x, [u / mu for u in ui], "-", color=c, alpha=.75, lw=1.4,
                   label=f"{k}")
        ut = m["u_txt"]; mt = mean(ut) or 1e-9
        ax[1].plot(x, [u / mt for u in ut], "-", color=c, alpha=.75, lw=1.4)
    ax[0].set_title("VISUAL pathway: image-token update u_img(l)/mean\n(higher early = image being 'encoded' in early layers)")
    ax[1].set_title("TEXT pathway: text-token update u_txt(l)/mean")
    for a in ax[:2]:
        a.set_xlabel("relative depth l/N"); a.axvspan(0, 0.25, color="gray", alpha=.10)
    ax[0].legend(fontsize=6.5, loc="upper right")
    # right: imgCoM vs funcCoM scatter
    for k, v in rows:
        if "func" not in v:
            continue
        c = FREE if v["family"] == "free" else ENC
        ax[2].plot(v["mat"]["img_com_rel"], v["func"]["com_rel"], "o", ms=11, color=c)
        ax[2].text(v["mat"]["img_com_rel"], v["func"]["com_rel"] + 0.012, k, fontsize=7, ha="center")
    ax[2].plot([0, 1], [0, 1], "k--", lw=.5, alpha=.4)
    ax[2].set_xlabel("image-update CoM (where image is WORKED ON)")
    ax[2].set_ylabel("functional fusion CoM (where image is USED)")
    ax[2].set_title("worked-on vs used depth")
    ax[2].set_xlim(0.3, 0.7); ax[2].set_ylim(0.2, 0.7)
    fig.tight_layout(); fig.savefig(OUT / "fig_maturation.png", dpi=130); print("wrote fig_maturation.png")


def fig_attention():
    files = {"LLaVA-1.5-7B": ("results_attn_llava_full.json", "llava-hf/llava-1.5-7b-hf", "enc"),
             "Qwen2.5-VL-7B": ("results_attn_qwen_full.json", "Qwen/Qwen2.5-VL-7B-Instruct", "enc"),
             "Gemma-4-12B": ("results_attn_gemma.json", "google/gemma-4-12B-it", "free"),
             "SAIL-7B": ("results_sail_attn.json", "SAIL-7B", "free")}
    bars = []
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    for name, (jf, key, fam) in files.items():
        p = ROOT / jf
        if not p.exists() or p.stat().st_size <= 2:
            continue
        d = json.loads(p.read_text())
        if key not in d:
            continue
        img = d[key]["last2img"]
        c = FREE if fam == "free" else ENC
        x = [(i + 1) / len(img) for i in range(len(img))]
        ax[1].plot(x, img, "-", color=c, lw=1.5, label=f"{name} ({mean(img):.2f})")
        bars.append((name, mean(img), c))
    # NEO from attn_mass (POPE)
    am = ROOT / "results_attn_mass.json"
    if am.exists() and am.stat().st_size > 2:
        d = json.loads(am.read_text())
        for st in ("SFT", "MT"):
            if st in d:
                cl = d[st]["classes"]; li_, vi = cl.index("last"), cl.index("vision")
                l2v = [d[st]["masses"][l][li_][vi] for l in range(len(d[st]["masses"]))]
                bars.append((f"NEO-2B-{st}*", mean(l2v), FREE))
    bars.sort(key=lambda b: b[1])
    ax[0].barh([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
    ax[0].axvspan(0.10, 0.30, color="green", alpha=.08)
    ax[0].axvspan(0.60, 0.80, color="red", alpha=.06)
    ax[0].set_xlabel("mean last-token -> image attention fraction")
    ax[0].set_title("FastV attention (blue=encoder-free, orange=encoder)\n"
                    "green=SAIL-paper modular 10-30%, red=their native 60-80%\n*NEO on POPE")
    ax[1].set_xlabel("relative depth l/N"); ax[1].set_ylabel("last->image fraction")
    ax[1].set_title("per-layer last->image"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_attention.png", dpi=130); print("wrote fig_attention.png")


if __name__ == "__main__":
    fig_fdi()
    fig_maturation()
    fig_attention()
