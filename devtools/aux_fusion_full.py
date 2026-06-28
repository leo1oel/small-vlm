"""Unified, ALL-LAYER fusion metrics for the encoder-free unified VLM.

Three layer-resolved curves, each with a swap-image null band, on VMCBench
dev-1000 (same prompt as the lmms-eval sweep, letter-logit scoring):

  cost(d)  cumulative decouplability: block EVERY text-query->vision-key edge
           in decoder layers [0..d); accuracy/NLL vs d. Null = same on a
           wrong (swapped) image. Functional-fusion onset L_f = first d whose
           damage exceeds the null band.

  nu(l)    single-layer causal necessity: KL( p_intact || p_cut@l ) where the
           cut removes the cross-modal edge at layer l ALONE. Redundancy makes
           this underestimate, but the curve localizes intensity; fusion
           center-of-mass CoM = sum l*nu(l) / sum nu(l). Null = same KL on a
           wrong image (nu_null).

  rho(l)   representational fusion rate: how much the answer-position hidden
           state at layer l depends on WHICH image is present:
              rho(l) = ||h_l(I) - h_l(I')|| / ||h_l(I) - h_l(no image)||.
           rho ~ 0 where text carries no image information, rises toward 1 as
           the image is mixed in. Representational onset = first l with rho
           above its low-layer floor.

Every layer 1..N is covered (N = num_hidden_layers). The expensive causal
sweeps (cost, nu, 4N forwards/sample) run on the first N_CAUSAL samples;
rho (3 forwards/sample) runs on all of them. Resume-safe jsonl append.

Usage: python devtools/aux_fusion_full.py <ckpt> <out.jsonl> [n] [n_causal]
Runs in the main .venv (transformers 5.10.2), eager attention.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm.data.dataset import (  # noqa: E402
    inject_query_placeholders,
    tokenizer_multimodal_token,
)
from vlm.inference.eval import (  # noqa: E402
    _data_args_from_config,
    build_prompt,
    load_model,
    prepare_media_inputs,
    resolve_conv_mode,
)

DEV = "cuda"
POST_PROMPT = "Answer with the option's letter from the given choices directly.\n"


def load_vmcbench():
    import datasets

    return datasets.load_dataset("suyc21/VMCBench", split="dev")


def doc_to_prompt(doc) -> str:
    op = "Options:\n" + "".join(f"{k}. {doc[k]}\n" for k in "ABCD")
    return f"Question: {doc['question']}\n{op}{POST_PROMPT}"


class FullProbe:
    def __init__(self, ckpt: str):
        self.model, self.processor, info = load_model(ckpt, bf16=True, attn_implementation="eager")
        # E4 (spec 2026-06-18): the probe runs on BOTH the encoder-free native
        # path and the matched CLIP-encoder arm (sft-clip) for the encoder-vs-
        # pixel R0 comparison — encode_images vs encode_raw_patches below.
        self.encoder_free = bool(info["encoder_free"])
        self.da = _data_args_from_config(self.model.config)
        self.conv = resolve_conv_mode(self.model.config, ckpt, conv_mode=None)
        self.tok = self.processor.tokenizer
        self.letter_ids = {c: self.tok(c, add_special_tokens=False).input_ids[0] for c in "ABCD"}
        self.layers = self.model.model.layers
        self.N = len(self.layers)

        # hook state
        self.mode = "none"  # none | cumulative | single
        self.depth = 0
        self.cut = -1
        self.capture = False
        self.vm = None
        self.txtpool = None
        self.cap: dict[int, tuple] = {}
        self._mc: dict[int, torch.Tensor] = {}
        for i, lyr in enumerate(self.layers):
            lyr.register_forward_pre_hook(self._pre(i), with_kwargs=True)
            lyr.register_forward_hook(self._post(i + 1))

    def _isolated(self, mask, dtype):
        key = id(mask)
        if key not in self._mc:
            n = int(self.vm.numel())
            if mask is None:
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
            m[0, :, tp[:, None], vp[None, :]] = neg
            self._mc[key] = m
        return self._mc[key]

    def _pre(self, i):
        def f(_m, args, kwargs):
            active = (self.mode == "cumulative" and i < self.depth) or (
                self.mode == "single" and i == self.cut
            )
            if active:
                kwargs = dict(kwargs)
                hidden = args[0] if args else kwargs["hidden_states"]
                kwargs["attention_mask"] = self._isolated(
                    kwargs.get("attention_mask"), hidden.dtype
                )
                return args, kwargs
            return None

        return f

    def _post(self, k):
        def f(_m, _a, out):
            if self.capture:
                h = (out[0] if isinstance(out, tuple) else out)[0]
                last = h[-1].float()
                mean = (
                    h[self.txtpool].float().mean(0)
                    if (self.txtpool is not None and bool(self.txtpool.any()))
                    else last
                )
                self.cap[k] = (last, mean)

        return f

    # ---- input builders --------------------------------------------------
    def _embed_with_image(self, doc, image):
        text = self.da.image_token + "\n" + doc_to_prompt(doc)
        # BREEN: inject one <query> per image at the trained placement, mirroring
        # generate_response, so the probe matches the trained/live input contract
        # (the splice expands <query> into the learnable query block). No-op for
        # non-BREEN checkpoints.
        if self.da.learnable_query_enabled:
            _turns = [{"from": "human", "value": text}]
            inject_query_placeholders(_turns, n_images=1, data_args=self.da)
            text = _turns[0]["value"]
        prompt = build_prompt(self.conv, text, self.da)
        ids = (
            tokenizer_multimodal_token(prompt, self.tok, self.da, return_tensors="pt")
            .unsqueeze(0)
            .to(DEV)
        )
        gk = prepare_media_inputs(self.model, self.processor, [image.convert("RGB")], [], self.da)
        if self.encoder_free:
            feats = self.model.encode_raw_patches(gk["images"], gk["image_position_ids"])
        else:
            feats, _ = self.model.encode_images(gk["images"])
        (_, _, am, _, emb, _, blk, _) = self.model.prepare_inputs_labels_for_multimodal(
            ids, None, torch.ones_like(ids), None, None, feats, None, with_image_block_ids=True
        )
        vm = (blk[0] >= 0).to(DEV)
        first_vis = int(vm.nonzero()[0])
        txtpool = (~vm).clone()
        txtpool[: first_vis + 1] = False
        return emb, am, vm, txtpool

    def _embed_text_only(self, doc):
        prompt = build_prompt(self.conv, doc_to_prompt(doc), self.da)
        ids = (
            tokenizer_multimodal_token(prompt, self.tok, self.da, return_tensors="pt")
            .unsqueeze(0)
            .to(DEV)
        )
        emb = self.model.get_input_embeddings()(ids)
        return emb, torch.ones(ids.shape, dtype=torch.long, device=DEV)

    # ---- forwards --------------------------------------------------------
    @torch.no_grad()
    def _logits_last(self, emb, am):
        return self.model(inputs_embeds=emb, attention_mask=am).logits[0, -1].float()

    def _score(self, lg, gt, letters):
        scores = {c: lg[self.letter_ids[c]].item() for c in letters}
        return dict(
            pred=max(scores, key=scores.get), nll=-lg.log_softmax(-1)[self.letter_ids[gt]].item()
        )

    @torch.no_grad()
    def capture_all(self, emb, am, vm, txtpool):
        self.mode, self.capture, self.vm, self.txtpool = "none", True, vm, txtpool
        self.cap = {}
        self._mc.clear()
        self.model(inputs_embeds=emb, attention_mask=am)
        self.capture = False
        return {k: v for k, v in self.cap.items()}

    # ---- per-sample driver ----------------------------------------------
    @torch.no_grad()
    def run(self, doc, donor, do_causal: bool):
        letters = [c for c in "ABCD" if str(doc[c]).strip() not in ("", "nan", "None")]
        gt = str(doc["answer"]).strip()
        if gt not in letters:
            return dict(skip="bad_gt")
        emb, am, vm, tp = self._embed_with_image(doc, doc["image"])
        s_emb, s_am, s_vm, s_tp = self._embed_with_image(doc, donor["image"])
        t_emb, t_am = self._embed_text_only(doc)
        rec = dict(
            gt=gt,
            letters=letters,
            category=doc["category"],
            seq_len=int(emb.shape[1]),
            n_vis=int(vm.sum()),
        )

        # rho: capture all-layer hidden states for I, I', no-image
        cap_I = self.capture_all(emb, am, vm, tp)
        cap_Ip = self.capture_all(s_emb, s_am, s_vm, s_tp)
        self.mode, self.capture, self.vm, self.txtpool = "none", True, None, None
        self.cap = {}
        self.model(inputs_embeds=t_emb, attention_mask=t_am)
        cap_0 = {k: v for k, v in self.cap.items()}
        self.capture = False
        dsw, dno, nrm = [], [], []
        for k in range(1, self.N + 1):
            hI, hIp, h0 = cap_I[k][0], cap_Ip[k][0], cap_0[k][0]
            dsw.append((hI - hIp).norm().item())
            dno.append((hI - h0).norm().item())
            nrm.append(hI.norm().item())
        rec["rho"] = dict(dswap=dsw, dnoimg=dno, norm=nrm)

        # intact / swap scores (depth 0)
        self.mode, self.vm = "none", vm
        self._mc.clear()
        lg_I = self._logits_last(emb, am)
        rec["intact"] = self._score(lg_I, gt, letters)
        self.vm = s_vm
        self._mc.clear()
        lg_Ip = self._logits_last(s_emb, s_am)
        rec["swap"] = self._score(lg_Ip, gt, letters)

        if not do_causal:
            return rec

        # cost(d): cumulative isolation, real image and swap-image null
        lp_I = lg_I.log_softmax(-1)
        lp_Ip = lg_Ip.log_softmax(-1)
        cost, cost_null, nu, nu_null = [], [], [], []
        self.mode = "cumulative"
        for d in range(1, self.N + 1):
            self.depth = d
            self.vm = vm
            self._mc.clear()
            cost.append(self._score(self._logits_last(emb, am), gt, letters))
            self.vm = s_vm
            self._mc.clear()
            cost_null.append(self._score(self._logits_last(s_emb, s_am), gt, letters))
        rec["cost"] = cost
        rec["cost_null"] = cost_null

        # nu(l): single-layer cut, KL vs the corresponding intact
        self.mode = "single"
        for l in range(self.N):
            self.cut = l
            self.vm = vm
            self._mc.clear()
            lq = self._logits_last(emb, am).log_softmax(-1)
            nu.append((lp_I.exp() * (lp_I - lq)).sum().item())
            self.vm = s_vm
            self._mc.clear()
            lqn = self._logits_last(s_emb, s_am).log_softmax(-1)
            nu_null.append((lp_Ip.exp() * (lp_Ip - lqn)).sum().item())
        rec["nu"] = nu
        rec["nu_null"] = nu_null
        self.mode = "none"
        return rec


@torch.no_grad()
def main():
    ckpt, out_path = sys.argv[1], Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    n_causal = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    ds = load_vmcbench()
    n = min(limit, len(ds)) if limit else len(ds)
    done = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    print(f"[full] ckpt={ckpt} n={n} n_causal={n_causal} resume={done}", flush=True)
    pr = FullProbe(ckpt)
    print(f"[full] N_layers={pr.N}", flush=True)
    f = open(out_path, "a")
    for i in range(done, n):
        doc = ds[i]
        donor = ds[(i + 37) % n]
        try:
            rec = pr.run(doc, donor, do_causal=(i < n_causal))
            rec["i"] = i
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rec = dict(i=i, skip="oom")
        except Exception as e:  # noqa: BLE001
            rec = dict(i=i, skip=f"{type(e).__name__}: {e}")
        f.write(json.dumps(rec) + "\n")
        if (i + 1) % 25 == 0:
            f.flush()
            print(f"[full] {i + 1}/{n}", flush=True)
    f.close()
    print("[full] done", flush=True)


if __name__ == "__main__":
    main()
