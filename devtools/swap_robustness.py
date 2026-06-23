"""Is R0 (the usable-image-signal null) sensitive to WHICH wrong image we use?

For each sample: intact prediction + K predictions each with a DIFFERENT wrong
image (donor offsets), same question. Reports accuracy with the right image vs
each wrong-image draw, so R0 = intact - swap can be checked for stability
across swap choices. Single forward each (d=0 only) -> cheap.

Runs in the MAIN .venv (bee-mix / unified VLM). Reuses aux_fusion_full.FullProbe.

Usage: python devtools/swap_robustness.py <ckpt> <out.json> [n] [k_swaps]
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aux_fusion_full import FullProbe, doc_to_prompt, load_vmcbench  # noqa: E402

DEV = "cuda"
OFFSETS = [37, 113, 251, 499, 661]  # distinct co-prime-ish donor offsets


@torch.no_grad()
def main():
    ckpt, out_path = sys.argv[1], Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    ds = load_vmcbench()
    n = min(n, len(ds))
    offs = OFFSETS[:k]
    pr = FullProbe(ckpt)
    print(f"[swaprob] ckpt={ckpt} n={n} k={k} offsets={offs}", flush=True)

    rows = []
    for i in range(n):
        doc = ds[i]
        gt = str(doc["answer"]).strip()
        if gt not in "ABCD":
            continue
        try:
            emb, am, vm, tp = pr._embed_with_image(doc, doc["image"])
            pr.mode, pr.vm = "none", vm
            pr._mc.clear()
            intact = pr._score(pr._logits_last(emb, am), gt, ["A", "B", "C", "D"])
            swaps = []
            for off in offs:
                donor = ds[(i + off) % n]
                se, sa, svm, _ = pr._embed_with_image(doc, donor["image"])
                pr.vm = svm
                pr._mc.clear()
                swaps.append(pr._score(pr._logits_last(se, sa), gt, ["A", "B", "C", "D"]))
            rows.append(dict(i=i, gt=gt, intact=intact["pred"],
                             swaps=[s["pred"] for s in swaps]))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(i=i, skip=f"{type(e).__name__}: {e}"))
        if (i + 1) % 50 == 0:
            print(f"[swaprob] {i + 1}/{n}", flush=True)

    ok = [r for r in rows if "skip" not in r]
    acc_intact = sum(r["intact"] == r["gt"] for r in ok) / len(ok)
    print(f"\n[swaprob] n_ok={len(ok)}  intact acc={acc_intact:.3f}", flush=True)
    swap_accs = []
    for j, off in enumerate(offs):
        a = sum(r["swaps"][j] == r["gt"] for r in ok) / len(ok)
        swap_accs.append(a)
        print(f"  swap offset +{off:>3}: acc={a:.3f}  R0={acc_intact - a:+.3f}", flush=True)
    import statistics as st
    print(f"  swap acc mean={st.mean(swap_accs):.3f} sd={st.pstdev(swap_accs):.4f}  "
          f"-> R0 range [{acc_intact - max(swap_accs):+.3f}, {acc_intact - min(swap_accs):+.3f}]",
          flush=True)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out[Path(ckpt).parent.name + "/" + Path(ckpt).name] = dict(
        intact=acc_intact, swap_accs=swap_accs, offsets=offs, n=len(ok))
    out_path.write_text(json.dumps(out, indent=1))
    print(f"[swaprob] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
