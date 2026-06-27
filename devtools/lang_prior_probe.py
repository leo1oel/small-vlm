"""Visual-gain decomposition: how much of the score is language prior vs vision?

Root-cause probe for "why is the language bias so severe / why doesn't vmc move".
For each checkpoint, run N samples under REAL image vs BLANK (all-zero) image and
report accuracy under each, plus the VISUAL GAIN = real_acc - blank_acc.

Interpretation:
  gain ~ 0  -> the benchmark is language-solvable; the model already gets its
               score from the text prior, vision adds nothing -> ROOT CAUSE is
               data/benchmark (H1), not the model. No loss/arch change can help.
  gain > 0  -> vision IS being used; compare across checkpoints (e.g. aim_pixel
               vs baseline) to see whether a recipe increases visual reliance.

Runs VMCBench (the vmc mystery) and, if loadable, POPE. Multiple ckpts in one
run for a clean side-by-side. Usage:
  python devtools/lang_prior_probe.py <ckptA>[,<ckptB>,...] [N]
"""

import sys

import torch
from PIL import Image

from vlm.inference.eval import generate_response, load_model

CKPTS = (
    sys.argv[1].split(",")
    if len(sys.argv) > 1
    else [
        "/gscratch/scrubbed/leoym/small-vlm-outputs/sft-unified-bee-mix/checkpoint-5000",
        "/gscratch/scrubbed/leoym/small-vlm-outputs/sft-unified-bee-mix-aimpixel/checkpoint-5000",
    ]
)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
VPOST = "Answer with the option's letter from the given choices directly.\n"
PPOST = "Answer the question using a single word or phrase.\n"


def vmc_samples(n):
    import datasets

    ds = datasets.load_dataset("suyc21/VMCBench", split="dev")
    out = []
    for i in range(min(n, len(ds))):
        d = ds[i]
        op = "Options:\n" + "".join(f"{k}. {d[k]}\n" for k in "ABCD")
        q = f"<image>\nQuestion: {d['question']}\n{op}{VPOST}"
        out.append((d["image"].convert("RGB"), q, str(d["answer"]).strip().upper()))
    return out


def pope_samples(n):
    """Best-effort POPE load; returns [] if not cacheable offline."""
    try:
        import datasets

        ds = datasets.load_dataset("lmms-lab/POPE", split="test")
        out = []
        for i in range(min(n, len(ds))):
            d = ds[i]
            q = f"<image>\n{d['question']}\n{PPOST}"
            out.append((d["image"].convert("RGB"), q, str(d["answer"]).strip().lower()))
        return out
    except Exception as e:
        print(f"[pope] skipped ({type(e).__name__}: {str(e)[:80]})", flush=True)
        return []


def vmc_letter(s):
    s = (s or "").strip().upper()
    return s[0] if s and s[0] in "ABCD" else "?"


def pope_yn(s):
    s = (s or "").strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    return s[:3]


def run_bench(model, processor, samples, norm, max_new):
    real_ok = blank_ok = flip = 0
    blank_yes = 0  # for POPE: how often blank-image answers 'yes'
    for img, q, gold in samples:
        blank = Image.new("RGB", img.size, 0)
        ar = norm(
            generate_response(
                model, processor, query=q, images=img, temperature=0.0, max_new_tokens=max_new
            )
        )
        ab = norm(
            generate_response(
                model, processor, query=q, images=blank, temperature=0.0, max_new_tokens=max_new
            )
        )
        real_ok += ar == gold
        blank_ok += ab == gold
        flip += ar != ab
        blank_yes += ab == "yes"
    n = len(samples)
    return real_ok / n, blank_ok / n, flip / n, blank_yes / n, n


def main():
    rows = []
    for CKPT in CKPTS:
        tag = (
            CKPT.rstrip("/").split("/")[-2].replace("sft-unified-bee-mix", "base")
            + "/"
            + CKPT.rstrip("/").split("/")[-1]
        )
        print(f"\n===== loading {tag} =====", flush=True)
        model, processor, _ = load_model(CKPT, bf16=True)
        model.eval()

        vmc = vmc_samples(N)
        vr, vb, vf, _, vn = run_bench(model, processor, vmc, vmc_letter, 8)
        print(
            f"[{tag}] VMC  real={vr:.3f} blank={vb:.3f} GAIN={vr - vb:+.3f} flip={vf:.3f} n={vn}",
            flush=True,
        )
        rows.append((tag, "VMC", vr, vb, vr - vb, vf))

        pope = pope_samples(N)
        if pope:
            pr, pb, pf, pby, pn = run_bench(model, processor, pope, pope_yn, 8)
            print(
                f"[{tag}] POPE real={pr:.3f} blank={pb:.3f} GAIN={pr - pb:+.3f} "
                f"flip={pf:.3f} blank_yes={pby:.3f} n={pn}",
                flush=True,
            )
            rows.append((tag, "POPE", pr, pb, pr - pb, pf))

        del model
        torch.cuda.empty_cache()

    print("\n========== VISUAL GAIN SUMMARY ==========", flush=True)
    print(f"{'ckpt':<28}{'bench':<6}{'real':>7}{'blank':>7}{'GAIN':>8}{'flip':>7}", flush=True)
    for tag, b, r, bl, g, f in rows:
        print(f"{tag:<28}{b:<6}{r:>7.3f}{bl:>7.3f}{g:>+8.3f}{f:>7.3f}", flush=True)
    print("\nGAIN~0 => benchmark language-solvable (root cause = data, H1).", flush=True)
    print("GAIN>0 and higher for a recipe => that recipe increases visual reliance.", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
