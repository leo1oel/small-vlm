"""Did aux-exit training change the fusion structure? (manipulation check)

Three probes on a checkpoint, over the same VMCBench dev-1000 the lmms-eval
sweeps used (suyc21/VMCBench, letter-logit scoring instead of generation):

  1. Isolation depth sweep: block every text-query -> vision-key edge in
     decoder layers [0..d) for d in DEPTHS; if aux training moved visual
     conditioning of text into layers <= 6, the aux checkpoint's accuracy/NLL
     curve must break earlier than the baseline's.
  2. Layer-k logit-lens readout (the exact aux-exit decode: shared final
     RMSNorm + tied lm_head on layer-k output) at the answer position, with
     the CORRECT image and with a SWAPPED image. Image-swap sensitivity of
     the layer-6 readout discriminates real early fusion from the two
     shortcuts (basis rotation / text-only answer-formatting): readability
     that survives the swap carries no image information.
  3. Per-layer attention mass from text queries (after the image block) to
     vision keys — did aux training rewire early-layer cross-modal attention?

Usage: python devtools/aux_fusion_probe.py <checkpoint_dir> <out.jsonl> [n]

Runs in the main .venv (transformers 5.10.2), eager attention so per-layer
attention_mask kwargs are materialized tensors (same hook technique as
neo_analysis/gemma4_vision_bench.py). Resume-safe: appends to <out.jsonl>,
skips already-done samples on restart.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm.inference.eval import (  # noqa: E402
    _data_args_from_config,
    build_prompt,
    load_model,
    prepare_media_inputs,
    resolve_conv_mode,
)

DEV = "cuda"
DEPTHS = [2, 4, 6, 8, 12, 16, 28]  # isolation: block text->vision in [0..d)
EXITS = [2, 4, 6, 8, 12, 16, 20, 24]  # logit-lens readout layers (28 = final)
ATTN_SEQ_CAP = 3000  # skip attention capture beyond this (memory guard)
POST_PROMPT = "Answer with the option's letter from the given choices directly.\n"


def load_vmcbench():
    import datasets

    return datasets.load_dataset("suyc21/VMCBench", split="dev")


def doc_to_prompt(doc) -> str:
    # Byte-identical to lmms_eval/tasks/vmcbench/utils.vmcbench_doc_to_text
    # with the default pre/post prompts.
    options_prompt = "Options:\n"
    for key in "ABCD":
        options_prompt += f"{key}. {doc[key]}\n"
    return f"Question: {doc['question']}\n{options_prompt}{POST_PROMPT}"


class ProbeRunner:
    def __init__(self, ckpt: str):
        self.model, self.processor, info = load_model(ckpt, bf16=True, attn_implementation="eager")
        assert info["encoder_free"], "probe assumes the encoder-free unified path"
        self.data_args = _data_args_from_config(self.model.config)
        self.conv_mode = resolve_conv_mode(self.model.config, ckpt, conv_mode=None)
        self.tok = self.processor.tokenizer
        self.letter_ids = {c: self.tok(c, add_special_tokens=False).input_ids[0] for c in "ABCD"}
        self.layers = self.model.model.layers
        self.final_norm = self.model.model.norm
        self.n_layers = len(self.layers)

        # --- isolation hook state (gemma4_vision_bench pattern) ---
        self.depth = 0
        self.vm = None  # (L,) bool, True on vision positions
        self._mask_cache: dict[int, torch.Tensor] = {}
        self._captured: dict[int, torch.Tensor] = {}
        self._capture_set: set[int] = set()
        for i, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(self._make_pre_hook(i), with_kwargs=True)
            layer.register_forward_hook(self._make_post_hook(i + 1))

    # -- hooks ------------------------------------------------------------
    def _isolated(self, mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
        key = id(mask)
        if key not in self._mask_cache:
            n = int(self.vm.numel())
            if mask is None:
                # eager fast path materialized no mask: build plain causal
                m = torch.zeros(1, 1, n, n, dtype=dtype, device=self.vm.device)
                m.masked_fill_(
                    torch.ones(n, n, dtype=torch.bool, device=self.vm.device).triu(1),
                    torch.finfo(dtype).min,
                )
            else:
                m = mask.clone()
            tp = (~self.vm).nonzero().squeeze(-1)
            vp = self.vm.nonzero().squeeze(-1)
            neg = False if m.dtype == torch.bool else torch.finfo(m.dtype).min
            m[0, :, tp[:, None], vp[None, :]] = neg  # text cannot read vision
            self._mask_cache[key] = m
        return self._mask_cache[key]

    def _make_pre_hook(self, i: int):
        def f(_mod, args, kwargs):
            if i < self.depth:
                kwargs = dict(kwargs)
                hidden = args[0] if args else kwargs["hidden_states"]
                kwargs["attention_mask"] = self._isolated(
                    kwargs.get("attention_mask"), hidden.dtype
                )
                return args, kwargs
            return None

        return f

    def _make_post_hook(self, k: int):
        def f(_mod, _args, out):
            if k in self._capture_set:
                h = out[0] if isinstance(out, tuple) else out
                self._captured[k] = h[:, -1:].detach()  # answer position only

        return f

    # -- input construction (mirrors generate_response, minus generation) --
    def build(self, doc, image):
        from vlm.data.dataset import inject_query_placeholders, tokenizer_multimodal_token

        text = self.data_args.image_token + "\n" + doc_to_prompt(doc)
        # BREEN: inject one <query> per image at the trained placement, mirroring
        # generate_response, so the probe matches the trained/live input contract
        # (the splice expands <query> into the learnable query block). No-op for
        # non-BREEN checkpoints.
        if self.data_args.learnable_query_enabled:
            _turns = [{"from": "human", "value": text}]
            inject_query_placeholders(_turns, n_images=1, data_args=self.data_args)
            text = _turns[0]["value"]
        prompt = build_prompt(self.conv_mode, text, self.data_args)

        input_ids = (
            tokenizer_multimodal_token(prompt, self.tok, self.data_args, return_tensors="pt")
            .unsqueeze(0)
            .to(DEV)
        )
        gen_kwargs = prepare_media_inputs(
            self.model, self.processor, [image.convert("RGB")], [], self.data_args
        )
        image_features = self.model.encode_raw_patches(
            gen_kwargs["images"], gen_kwargs["image_position_ids"]
        )
        (_, _, attention_mask, _, inputs_embeds, _, block_ids, _) = (
            self.model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                torch.ones_like(input_ids),
                None,
                None,
                image_features,
                None,
                with_image_block_ids=True,
            )
        )
        vm = (block_ids[0] >= 0).to(DEV)
        return inputs_embeds, attention_mask, vm

    # -- scoring ------------------------------------------------------------
    def _score(self, logits_last: torch.Tensor, gt: str, letters: list[str]):
        lg = logits_last.float()
        logprobs = lg.log_softmax(-1)
        scores = {c: lg[self.letter_ids[c]].item() for c in letters}
        return dict(
            pred=max(scores, key=scores.get),
            nll=-logprobs[self.letter_ids[gt]].item(),
        )

    @torch.no_grad()
    def forward_scored(
        self, embeds, mask2d, vm, gt, letters, depth=0, capture=False, want_attn=False
    ):
        self.depth = depth
        self.vm = vm
        self._mask_cache.clear()
        self._captured.clear()
        self._capture_set = set(EXITS) if capture else set()
        out = self.model(
            inputs_embeds=embeds,
            attention_mask=mask2d,
            output_attentions=want_attn,
        )
        rec = self._score(out.logits[0, -1], gt, letters)
        if capture:
            exits = {}
            for k, h in self._captured.items():
                lg = self.model.lm_head(self.final_norm(h[0]))[0]
                exits[k] = self._score(lg, gt, letters)
            rec["exits"] = exits
        if want_attn and out.attentions is not None:
            # text-after-image queries -> vision keys, mean over heads+queries
            first_vis = int(vm.nonzero()[0])
            tq = (~vm).clone()
            tq[: first_vis + 1] = False  # only text that CAN see the image
            t2v, last2v = [], []
            for a in out.attentions:  # (1, H, L, L)
                a = a[0].float()
                t2v.append(a[:, tq][:, :, vm].sum(-1).mean().item())
                last2v.append(a[:, -1][:, vm].sum(-1).mean().item())
            rec["attn_t2v"] = t2v
            rec["attn_last2v"] = last2v
        return rec


@torch.no_grad()
def main():
    ckpt, out_path = sys.argv[1], Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ds = load_vmcbench()
    n = min(limit, len(ds)) if limit else len(ds)

    done = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    print(f"[probe] ckpt={ckpt} n={n} resume_from={done}", flush=True)
    runner = ProbeRunner(ckpt)

    f = open(out_path, "a")
    for i in range(done, n):
        doc = ds[i]
        donor = ds[(i + 37) % n]  # fixed derangement-ish swap pairing
        letters = [c for c in "ABCD" if str(doc[c]).strip() not in ("", "nan", "None")]
        gt = str(doc["answer"]).strip()
        if gt not in letters:
            f.write(json.dumps(dict(i=i, skip="bad_gt")) + "\n")
            continue
        try:
            embeds, mask2d, vm = runner.build(doc, doc["image"])
            seq_len = int(embeds.shape[1])
            rec = dict(
                i=i,
                gt=gt,
                letters=letters,
                category=doc["category"],
                seq_len=seq_len,
                n_vis=int(vm.sum()),
            )
            rec["intact"] = runner.forward_scored(
                embeds,
                mask2d,
                vm,
                gt,
                letters,
                depth=0,
                capture=True,
                want_attn=seq_len <= ATTN_SEQ_CAP,
            )
            for d in DEPTHS:
                rec[f"iso_d{d}"] = runner.forward_scored(embeds, mask2d, vm, gt, letters, depth=d)
            s_embeds, s_mask2d, s_vm = runner.build(doc, donor["image"])
            rec["swap"] = runner.forward_scored(
                s_embeds, s_mask2d, s_vm, gt, letters, depth=0, capture=True
            )
            f.write(json.dumps(rec) + "\n")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            f.write(json.dumps(dict(i=i, skip="oom")) + "\n")
        except Exception as e:  # noqa: BLE001 — log and continue, requeue-safe
            f.write(json.dumps(dict(i=i, skip=f"{type(e).__name__}: {e}")) + "\n")
        if (i + 1) % 25 == 0:
            f.flush()
            print(f"[probe] {i + 1}/{n}", flush=True)
    f.close()

    # quick aggregate
    recs = [json.loads(line) for line in open(out_path)]
    ok = [r for r in recs if "skip" not in r]
    print(f"[probe] scored {len(ok)}/{len(recs)}", flush=True)

    def acc(key):
        sel = [r[key]["pred"] == r["gt"] for r in ok if key in r]
        return sum(sel) / max(len(sel), 1)

    msg = f"intact={acc('intact'):.3f} swap={acc('swap'):.3f}"
    for d in DEPTHS:
        msg += f" d{d}={acc(f'iso_d{d}'):.3f}"
    print(f"[probe] {msg}", flush=True)


if __name__ == "__main__":
    main()
