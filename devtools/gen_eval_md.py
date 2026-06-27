"""Regenerate logs/eval_results.md from all lmms-eval result JSONs.

Reads logs/lmms_eval/<run>__checkpoint-<step>/<ts>_results.json (latest
timestamp per (run,step,task) wins) and writes a grouped VMCBench / MMVP /
POPE markdown summary. Idempotent — run any time to refresh.
"""

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ORDER = [
    "sft-clip",
    "sft-clip-large",
    "sft-unified",
    "sft-unified-bee-mix",
    "sft-unified-bee-mix-aux",
    "sft-unified-bee-mix-aimpixel",
    "sft-unified-bee-mix-nepa",
    "sft-unified-staged",
]


def rk(r: str) -> int:
    return RUN_ORDER.index(r) if r in RUN_ORDER else 99


def main() -> None:
    tab: dict[tuple, dict] = defaultdict(dict)
    for rf in sorted(glob.glob(str(ROOT / "logs/lmms_eval/sft-*__checkpoint-*/*_results.json"))):
        m = re.match(r".*/(.+)__checkpoint-(\d+)/", rf)
        run, step = m.group(1), int(m.group(2))
        try:
            data = json.load(open(rf))
        except Exception:
            continue
        for task, mets in (data.get("results") or {}).items():
            for k, v in mets.items():
                if (
                    isinstance(v, (int, float))
                    and not k.endswith("_stderr")
                    and not k.startswith("alias")
                ):
                    tab[(run, step, task)][k.split(",")[0]] = v

    def pct(d, k):
        return f"{d[k] * 100:.1f}" if k in d else "–"

    o = [
        "# small-vlm 评测结果汇总 (VMCBench / MMVP / POPE)",
        "",
        "_随机基线: VMCBench 25% · MMVP 50%(单图)/0%(pair) · POPE 50%。`devtools/gen_eval_md.py` 自动生成。_",
        "",
        "## VMCBench dev (1000)",
        "",
        "| run | step | **average** | doc | general | ocr | reason |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for run, step, _ in sorted(
        [k for k in tab if k[2] == "vmcbench"], key=lambda x: (rk(x[0]), x[1])
    ):
        d = tab[(run, step, "vmcbench")]
        o.append(
            f"| {run} | {step} | **{pct(d, 'average')}** | {pct(d, 'doc')} | "
            f"{pct(d, 'general')} | {pct(d, 'ocr')} | {pct(d, 'reason')} |"
        )

    o += ["", "## MMVP (300)", "", "| run | step | accuracy | **pair_acc** |", "|---|--:|--:|--:|"]
    for run, step, _ in sorted([k for k in tab if k[2] == "mmvp"], key=lambda x: (rk(x[0]), x[1])):
        d = tab[(run, step, "mmvp")]
        o.append(
            f"| {run} | {step} | {pct(d, 'mmvp_accuracy')} | **{pct(d, 'mmvp_pair_accuracy')}** |"
        )

    o += [
        "",
        "## POPE (9000)",
        "",
        "_recall→100 且 precision≈50 = 退化为恒答 yes(健康 recall≈precision≈accuracy)_",
        "",
        "| run | step | **accuracy** | f1 | precision | recall |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for run, step, _ in sorted([k for k in tab if k[2] == "pope"], key=lambda x: (rk(x[0]), x[1])):
        d = tab[(run, step, "pope")]
        o.append(
            f"| {run} | {step} | **{pct(d, 'pope_accuracy')}** | {pct(d, 'pope_f1_score')} | "
            f"{pct(d, 'pope_precision')} | {pct(d, 'pope_recall')} |"
        )

    out = ROOT / "logs" / "eval_results.md"
    out.write_text("\n".join(o) + "\n")
    n = len({(r, s) for (r, s, t) in tab})
    print(f"wrote {out} ({n} checkpoints)")


if __name__ == "__main__":
    main()
