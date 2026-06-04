# small-vlm 审计修复、CLIP 共训练移除与依赖全面升级实施计划（v2）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ① 修复审计确认的 bug（zero2 差分学习率丢失、硬编码路径）；② 整体删除 CLIP 共训练相关代码（含 open_clip / webdataset / unified dataset / multitask trainer / dual_task）；③ 落地快速优化（kernels、fused AdamW、dtype 清理、tokenizer 缓存、视觉塔 sdpa 钉死）；④ 全面升级依赖至最新（transformers 5.10.1、torch / accelerate / deepspeed / datasets 最新兼容版）并完成 v5 迁移。

**Architecture:** 任务按依赖排序：先删除（缩小后续所有任务的修改面），再修复/优化（全部在旧 pin 上完成并验证，保证升级前是干净基线），最后升级（出问题时来源清晰）。Task 1-8 相互独立可删；Task 9（升级）依赖前面任务完成。

**Tech Stack:** Python 3.13 / uv / hydra / pytest；升级目标 transformers==5.10.1、torch 最新（与 deepspeed 兼容为准）、accelerate>=1.13、deepspeed>=0.19、datasets>=4.8

______________________________________________________________________

## 执行约定（每个任务必须遵守）

**质量要求（用户明确指示）：** 每个改动在实施前要认真核实细节正确性，实施后必须经过独立审查 agent 审查，参考网上资料和 `.venv` 内安装包源码确认 API 用法正确，确保不引入 bug。

**每个任务的标准流程:**

1. 实施 subagent 按步骤执行（含运行验证命令）
1. 实施完成后、commit 之前，dispatch 一个**审查 agent**，prompt 模板如下（替换 `{TASK}` 为任务标题、`{REQUIREMENTS}` 为该任务的"背景"段落）：

```
你是代码审查员。审查 /mmfs1/gscratch/krishna/leoym/small-vlm 工作区中未提交的改动（git diff + git status）。
本次改动的目标：{TASK} — {REQUIREMENTS}
要求：
1. 逐行阅读 diff，对照目标检查正确性与完整性（有没有漏改的引用、删了一半的符号、破坏的 import）。
2. 对 diff 中用到的每个第三方 API（transformers/torch/deepspeed/hydra/...），打开 .venv/lib/python3.13/site-packages/ 下的实际源码核对签名与行为；必要时用 ToolSearch 加载 WebSearch/Context7 查官方文档核对。
3. 运行 `uv run python -c "import vlm"`、`uv run pytest tests/ -x -q`、`uv run python devtools/lint.py` 确认无回归。
4. 输出：APPROVE 或 REJECT + 逐条问题清单（文件:行号 + 修复建议）。对不确定的 API 行为必须给出源码/文档依据，不允许凭记忆断言。
```

1. 审查 REJECT → 实施 agent 修复 → 重新审查，直到 APPROVE
1. APPROVE 后才 commit

______________________________________________________________________

## Setup

- [ ] **Step 0.1: 创建工作分支**

```bash
cd /mmfs1/gscratch/krishna/leoym/small-vlm
git checkout -b audit-fixes
```

______________________________________________________________________

### Task 1: 删除 zero2.json 的 optimizer block（恢复差分学习率）

**背景:** transformers 的 DeepSpeed 集成（`integrations/deepspeed.py:371-389`）在 DS config 含 `optimizer` block 时会用 DummyOptim **跳过** `VLMTrainer.create_optimizer()`，导致 `optimizer.py` 里 vision/connector/LLM 的差分学习率参数组被静默丢弃。zero3.json 没有此 block（正确写法），zero2.json 有（pretrain/train 配置在用，受害最深）。

**Files:**

- Modify: `src/vlm/config/deepspeed/zero2.json:13-21`

- Modify: `src/vlm/config/deepspeed/zero3_offload.json`（同样的 block，目前是死配置，顺手清理）

- [ ] **Step 1.1: 删除 zero2.json 的 optimizer block**

删除第 13-21 行整个 `"optimizer": {...}` 键。修改后的完整文件：

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": false,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

- [ ] **Step 1.2: 同样删除 zero3_offload.json 里的 optimizer block**（先 Read 该文件确认 block 位置，只删 `"optimizer"` 键，其余保持不变）

- [ ] **Step 1.3: 验证两个 json 合法且不含 optimizer 键**

```bash
uv run python -c "
import json
for f in ['src/vlm/config/deepspeed/zero2.json', 'src/vlm/config/deepspeed/zero3.json', 'src/vlm/config/deepspeed/zero3_offload.json']:
    d = json.load(open(f))
    assert 'optimizer' not in d, f'{f} still has optimizer block'
    print(f, 'OK')
"
```

Expected: 三行 `OK`

- [ ] **Step 1.4: 审查（按"执行约定"流程）→ Commit**

```bash
git add src/vlm/config/deepspeed/zero2.json src/vlm/config/deepspeed/zero3_offload.json
git commit -m "fix: remove DS-native optimizer block so custom per-component LR groups take effect"
```

______________________________________________________________________

### Task 2: 整体删除 CLIP 共训练代码

**背景:** 用户决定不再需要 CLIP 共训练（commit d7d4bac 引入）。删除面已通过 grep 全量确认：`open_clip: True` 没有任何 yaml 在用；`webdataset` 仅被 unified_dataset 使用；tests/ 无相关引用。删除范围 = task_modes/clip\_\* 前向分支、open_clip 加载、dual_task、UnifiedDataset、MultiTaskTrainer、相关配置与依赖。保留：`llama3.2-3b-clip/dino`、`*-cc12m` 等模型 yaml（它们只是视觉编码器替换实验，走 AutoModel 加载，不涉及共训练）、`train-llama-clip.yaml`/`train-llama-dino.yaml`（执行时核实其内容确实不引用 unified/dual_task 后保留）。

**Files:**

- Delete: `src/vlm/data/unified_dataset.py`, `src/vlm/train/multitask_trainer.py`

- Delete: `src/vlm/config/unified-pretrain-llava.yaml`, `unified-pretrain-qwen.yaml`, `unified-finetune-llava.yaml`, `unified-finetune-qwen.yaml`

- Delete: `src/vlm/config/trainer/finetune-unified.yaml`, `src/vlm/config/dataset/unified-pretrain.yaml`, `src/vlm/config/dataset/unified-finetune.yaml`

- Delete: `src/vlm/config/model/llava-7b-unified.yaml`, `src/vlm/config/model/qwen2.5-3b-unified.yaml`

- Delete: `src/vlm/config/trainer/learning_rate/llava-unified.yaml`（删除前 grep 确认无引用）

- Modify: `src/vlm/data/__init__.py`, `src/vlm/data/data_arguments.py`, `src/vlm/models/modeling_vlm.py`, `src/vlm/models/processing_vlm.py`, `src/vlm/models/configuration_vlm.py`, `src/vlm/train/vlm_trainer.py`, `src/vlm/vlm.py`, `src/vlm/config/config_schema.py`, `pyproject.toml`

- [ ] **Step 2.1: 删除文件**

（含 `modeling_vlm_old.py`：经与用户确认，它是缺少 use_all_tokens/SigLIP-pooler 特性的更早版本，直接删除，CLIP 移除在现行 `modeling_vlm.py` 上做。）

```bash
git rm src/vlm/models/modeling_vlm_old.py
git rm src/vlm/data/unified_dataset.py src/vlm/train/multitask_trainer.py
git rm src/vlm/config/unified-pretrain-llava.yaml src/vlm/config/unified-pretrain-qwen.yaml \
       src/vlm/config/unified-finetune-llava.yaml src/vlm/config/unified-finetune-qwen.yaml
git rm src/vlm/config/trainer/finetune-unified.yaml \
       src/vlm/config/dataset/unified-pretrain.yaml src/vlm/config/dataset/unified-finetune.yaml
git rm src/vlm/config/model/llava-7b-unified.yaml src/vlm/config/model/qwen2.5-3b-unified.yaml
grep -rn "learning_rate/llava-unified\|llava-unified" src/vlm/config --include="*.yaml" || git rm src/vlm/config/trainer/learning_rate/llava-unified.yaml
```

- [ ] **Step 2.2: data/**init**.py 改为**

```python
from .data_arguments import DataArguments, get_data_args
from .dataset import make_supervised_data_module
from .sampler import MultiModalLengthGroupedSampler

__all__ = [
    "get_data_args",
    "DataArguments",
    "make_supervised_data_module",
    "MultiModalLengthGroupedSampler",
]
```

- [ ] **Step 2.3: data_arguments.py 删除 clip 字段**

`DataArguments` 中删除以下字段：`clip_data_path`、`clip_image_folder`、`clip_webdataset_urls`、`clip_data_type`、`clip_dataset_size`、`vlm_batch_size`、`clip_batch_size`。`get_data_args` 中删除对应的 7 个 kwarg。

- [ ] **Step 2.4: config_schema.py 删除字段**

- `DatasetConfig`：删除 `clip_data_type`、`clip_dataset_size`、`clip_data_path`、`clip_image_folder`、`clip_webdataset_urls`、`vlm_batch_size`、`clip_batch_size`

- `ModelConfig`：删除 `dual_task: bool = False`

- `VisualEncoderConfig`：删除 `open_clip: bool = False`、`open_clip_model: str | None = None`

- [ ] **Step 2.5: modeling_vlm.py 清理（最大的一处）**

逐项：

1. `forward`（130-225 行）：签名删除 `task_modes`、`clip_input_ids`、`clip_attention_mask`、`clip_images` 四个参数；删除整个 `if task_modes is not None and "clip" in task_modes:` 分支（154-201 行）。保留分支后的常规路径（203-225 行）不动。
1. `encode_images`（340-363 行）与 `encode_images_raw`（285-338 行）：签名删除 `input_ids`、`attention_mask` 参数；`encode_images_raw` 删除整个 `if input_ids is not None:` 分支（292-302 行）和 `if self.config.vision_config.open_clip:` 分支（304-313 行）及 `if self.config.dual_task:` 分支（315-319 行），收敛为单一路径：

```python
def encode_images_raw(self: Any, images: Tensor) -> tuple[list[Tensor] | Tensor, Any]:
    """Encode images using vision model only, without connector."""
    outputs = self.model.vision_model(
        images,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[self.config.vision_config.output_layer].to(
        images.dtype
    )
    if self.config.vision_config.use_all_tokens:
        image_features = hidden_states
    elif self.config.vision_config.use_cls_token:
        if "siglip" in self.config.vision_config.hf_name:
            image_features = outputs.pooler_output.unsqueeze(1)
        else:
            image_features = hidden_states[:, 0:1]
    else:
        image_features = hidden_states[:, 1:]
    image_features = self.model.connector(image_features)
    return image_features, outputs
```

1. `_build_vision_model`（73-88 行）：删除 open_clip 分支，收敛为：

```python
def _build_vision_model(self: Any, config: Any) -> PreTrainedModel:
    vision_config = config.vision_config
    visual_encoder: PreTrainedModel = AutoModel.from_pretrained(
        vision_config.hf_name, trust_remote_code=True
    )
    if getattr(visual_encoder, "vision_model", None):
        visual_encoder = visual_encoder.vision_model  # pyright: ignore
    return visual_encoder
```

1. 类工厂 dict（599-614 行）：删除 `"supports_report_metrics": True,` 一行。
1. 文件内搜索 `report_metrics`、`open_clip`、`dual_task`、`task_modes`、`clip_`，确认零残留。

- [ ] **Step 2.6: processing_vlm.py 删除 open_clip**

`from_names` 等方法签名中删除 `open_clip_model` 参数（38、48 行），删除 61-64 行的 `if open_clip_model:` 分支（`import open_clip` / `create_model_and_transforms`）。同步修改 `vlm.py:102-110` 的调用处（删除第三个实参 `model_cfg.visual_encoder.open_clip_model`）。

- [ ] **Step 2.7: configuration_vlm.py 删除 dual_task**

`DynamicVLMConfig.__init__`（50-66 行）：删除 `dual_task: bool = False` 参数和 `self.dual_task: bool = dual_task` 赋值。

- [ ] **Step 2.8: vlm.py 清理**

1. `load_model` else 分支（81-95 行）：删除 `if model_cfg.visual_encoder.open_clip:` 整个 if/else，只保留原 else 内容（AutoConfig 路径）：

```python
        hf_config = AutoConfig.from_pretrained(model_cfg.visual_encoder.hf_name)
        if getattr(hf_config, "vision_config", None):
            hf_config = hf_config.vision_config
        vision_config_dict = hf_config if isinstance(hf_config, dict) else hf_config.to_dict()
```

1. `VLMConfig(...)` 构造（113-119 行）：删除 `dual_task=model_cfg.dual_task,` 一行。
1. `vlm()` 函数（156-172 行）：删除 `if cfg.model.dual_task:` 两处分支（日志分支和 data_module 分支），收敛为始终：

```python
        model, processor = load_model(cfg.model, cfg.trainer)
        model.to(training_args.device)
        data_args = get_data_args(cfg.dataset, cfg.model)
        log.info("Creating data module")
        data_module = make_supervised_data_module(processor=processor, data_args=data_args)
        train(model, training_args, data_module, processor)
```

1. 删除顶部不再使用的 `AutoProcessor` import（若仅 clip_tokenizer 在用）。

- [ ] **Step 2.9: vlm_trainer.py 脱离 MultiTaskTrainer**

```python
from transformers.trainer import Trainer, has_length, is_datasets_available
```

`class VLMTrainer(MultiTaskTrainer):` → `class VLMTrainer(Trainer):`，删除 `from .multitask_trainer import MultiTaskTrainer`。

- [ ] **Step 2.10: pyproject.toml 删除依赖**

从 `dependencies` 删除 `"open-clip-torch>=3.2.0"` 和 `"webdataset>=1.0.2"`，然后：

```bash
uv sync --all-extras --dev
```

- [ ] **Step 2.11: 零残留验证**

```bash
grep -rn "open_clip\|webdataset\|dual_task\|task_modes\|clip_images\|clip_input_ids\|clip_attention_mask\|clip_batch_size\|clip_tokenizer\|MultiTaskTrainer\|report_metrics\|make_unified\|UnifiedData" src/ tests/ && echo "RESIDUE FOUND" || echo "clean"
uv run python -c "import vlm; from vlm.data import make_supervised_data_module; from vlm.train.vlm_trainer import VLMTrainer; print('import OK')"
uv run pytest tests/ -q
uv run python devtools/lint.py
```

Expected: `clean` + `import OK` + 测试通过 + lint 干净

- [ ] **Step 2.12: 审查（按"执行约定"流程，重点：grep 验证零残留、确认保留的 yaml 不引用已删符号）→ Commit**

```bash
git add -A
git commit -m "refactor: remove CLIP co-training (open_clip, webdataset, unified dataset, multitask trainer, dual_task)"
```

______________________________________________________________________

### Task 3: 路径参数化 + 适配本集群的 train.slurm

**背景:** `/pasteur`、`/pasteur2` 在当前文件系统不存在。策略：deepspeed json 路径改为代码内相对包目录解析；数据/模型路径用 `${oc.env:...}`。SLURM 脚本按本集群（hyak）实际作业风格重写（参考用户提供的 bagel 项目脚本：`partition=ckpt-all`、`account=cse-ckpt`、`gpus=l40:4`、`--requeue`、单节点）。**requeue 意味着作业会被抢占重排，必须支持自动断点续训**，因此本任务同时加 auto-resume 逻辑。

**环境变量约定:** `VLM_DATA_ROOT`（数据根）、`VLM_MODEL_ROOT`（本地编码器权重根）、`VLM_PRETRAIN_CKPT`（pretrain 产出 checkpoint，finetune 用，可不设）。

**Files:**

- Modify: `src/vlm/train/training_arguments.py`（deepspeed 解析）

- Modify: `src/vlm/config/trainer/finetune.yaml`, `pretrain.yaml`, `train.yaml`

- Modify: `src/vlm/config/dataset/llava-pretrain.yaml`, `llava-finetune.yaml`

- Modify: `src/vlm/config/model/llava-7b-cc12m.yaml`, `qwen2.5-7b-cc12m.yaml`, `llama3.2-3b-clip.yaml`, `llama3.2-3b-dino.yaml`

- Modify: `src/vlm/train/train.py`（auto-resume）

- Modify: `train.slurm`, `README.md`

- [ ] **Step 3a.1: deepspeed 路径代码内解析**

`src/vlm/train/training_arguments.py` 顶部补充 `from pathlib import Path`（若缺失），在 `get_training_args` 之前加入：

```python
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _resolve_deepspeed(ds: str | None) -> str | None:
    """Resolve a deepspeed config: absolute path as-is; bare filename relative to the
    packaged config/deepspeed/ directory. Keeps yaml configs host-independent."""
    if ds is None:
        return None
    p = Path(ds)
    if p.is_file():
        return str(p)
    candidate = _CONFIG_DIR / "deepspeed" / p.name
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        f"DeepSpeed config not found: {ds} (also tried {candidate})"
    )
```

`get_training_args` 里 `deepspeed=config.deepspeed,` 改为 `deepspeed=_resolve_deepspeed(config.deepspeed),`。

- [ ] **Step 3a.2: trainer yaml 的 deepspeed 行改为裸文件名**

- `finetune.yaml:2` → `deepspeed: zero3.json`

- `pretrain.yaml:2` → `deepspeed: zero2.json`

- `train.yaml:2` → `deepspeed: zero2.json`

- [ ] **Step 3a.3: 验证解析函数**

```bash
uv run python -c "
from vlm.train.training_arguments import _resolve_deepspeed
p = _resolve_deepspeed('zero3.json')
print(p)
assert p.endswith('src/vlm/config/deepspeed/zero3.json')
assert _resolve_deepspeed(None) is None
try:
    _resolve_deepspeed('nonexistent.json')
    raise AssertionError('should have raised')
except FileNotFoundError:
    print('FileNotFoundError OK')
"
```

- [ ] **Step 3b.1: dataset yaml 参数化**

`src/vlm/config/dataset/llava-pretrain.yaml`：

```yaml
name: llava-pretrain
path: ${oc.env:VLM_DATA_ROOT}/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json
type: json
lazy_preprocess: True
is_multimodal: True
image_folder: ${oc.env:VLM_DATA_ROOT}/LLaVA-Pretrain/images
```

`src/vlm/config/dataset/llava-finetune.yaml`：

```yaml
name: llava-finetune
path: ${oc.env:VLM_DATA_ROOT}/LLaVA-Instruct-150K/llava_v1_5_mix665k.json
type: json
lazy_preprocess: True
is_multimodal: True
image_folder: ${oc.env:VLM_DATA_ROOT}/llava-images
image_aspect_ratio: pad
```

（子目录名按原布局起的合理名字；数据实际落位不同则按实际调整，数据需另行下载。）

- [ ] **Step 3c.1: trainer yaml 的 from_pretrained 参数化**

- `finetune.yaml`: `from_pretrained: /pasteur2/...` → `from_pretrained: ${oc.env:VLM_PRETRAIN_CKPT,null}`

- `train.yaml`: 删除已失效的 `from_pretrained: /pasteur/.../checkpoint-8720` 行，加注释 `# from_pretrained: set via trainer.from_pretrained=... or VLM_PRETRAIN_CKPT`

- [ ] **Step 3c.2: requeue 自动续训支持**

(1) `src/vlm/train/train.py` 把：

```python
if training_args.resume_from_checkpoint:
    log.info(f"Resuming from checkpoint: {training_args.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
else:
    log.info("Training without resuming from checkpoint")
    trainer.train()
```

改为：

```python
if training_args.resume_from_checkpoint:
    log.info(f"Resuming from checkpoint: {training_args.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
else:
    from transformers.trainer_utils import get_last_checkpoint

    last_ckpt = get_last_checkpoint(training_args.output_dir)
    if last_ckpt is not None:
        log.info(f"Auto-resuming from last checkpoint: {last_ckpt} (requeued job?)")
    else:
        log.info("Training from scratch (no checkpoint in output_dir)")
    trainer.train(resume_from_checkpoint=last_ckpt)
```

(2) `finetune.yaml` 把 `save_only_model: True` 改为 `save_only_model: False` 并加注释 `# requeue/auto-resume needs optimizer state in checkpoints`。

(3) 注意（写入计划执行说明）：hydra run dir 含时间戳（`config.yaml` 的 `hydra.run.dir`），requeue 后会进新目录导致找不到旧 checkpoint —— 所以 slurm 提交时必须用稳定的 `trainer.output_dir` 覆盖（见 Step 3e.1 的启动命令）。

- [ ] **Step 3d.1: model yaml 的本地视觉编码器路径参数化**

- `llava-7b-cc12m.yaml` / `qwen2.5-7b-cc12m.yaml`: `hf_name: ${oc.env:VLM_MODEL_ROOT}/open_clip/ViT-L-14-hf`

- `llama3.2-3b-clip.yaml`: `hf_name: ${oc.env:VLM_MODEL_ROOT}/clip_datacomp_s`

- `llama3.2-3b-dino.yaml`: `hf_name: ${oc.env:VLM_MODEL_ROOT}/dino_datacomp10m`

- [ ] **Step 3e.1: 重写 train.slurm（hyak ckpt-all / l40:4 / requeue）**

```bash
#!/bin/bash
#SBATCH --job-name=small-vlm
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log

set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
mkdir -p logs
source .venv/bin/activate

# Host-specific roots (adjust to where data/models actually live)
export VLM_DATA_ROOT="${VLM_DATA_ROOT:-/gscratch/krishna/${USER}/data}"
export VLM_MODEL_ROOT="${VLM_MODEL_ROOT:-/gscratch/krishna/${USER}/models}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NGPUS=4
export MASTER_PORT=29650
CONFIG_NAME="${CONFIG_NAME:-finetune-llava}"
# Stable output dir so --requeue auto-resume can find previous checkpoints
# (hydra's default run dir is timestamped and changes on every requeue).
RUN_DIR="${PROJECT_ROOT}/outputs/${CONFIG_NAME}"

echo "[small-vlm] node=$(hostname) config=${CONFIG_NAME} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true

python -u -m torch.distributed.run \
    --standalone \
    --nproc_per_node "${NGPUS}" \
    -m vlm -cn "${CONFIG_NAME}" \
    trainer.output_dir="${RUN_DIR}" \
    hydra.run.dir="${RUN_DIR}/hydra"

echo "END TIME: $(date)"
```

注意：`--account=cse-ckpt` 来自用户参考脚本；执行时跑 `hyakalloc` 确认当前用户在该账号下有 ckpt 额度，若无则换成 `hyakalloc` 列出的实际账号（如 krishna 的 ckpt 关联账号）。

- [ ] **Step 3.6: 验证 hydra 配置可解析（不实际训练）**

```bash
VLM_DATA_ROOT=/tmp VLM_MODEL_ROOT=/tmp uv run python -m vlm -cn finetune-llava --cfg job | head -40
VLM_DATA_ROOT=/tmp VLM_MODEL_ROOT=/tmp uv run python -m vlm -cn pretrain-llava --cfg job > /dev/null && echo "config OK"
```

Expected: 解析后的配置正常打印（`deepspeed: zero3.json`、`path: /tmp/...`），无 InterpolationResolutionError

- [ ] **Step 3.7: README 增加环境变量说明**

在 README.md 训练命令之前插入：

```markdown
### Environment variables

| Variable | Meaning | Example |
|---|---|---|
| `VLM_DATA_ROOT` | Root directory for datasets | `/gscratch/krishna/$USER/data` |
| `VLM_MODEL_ROOT` | Root for locally stored vision encoders | `/gscratch/krishna/$USER/models` |
| `VLM_PRETRAIN_CKPT` | Stage-1 checkpoint consumed by finetune configs (optional) | `outputs/.../checkpoint-8000` |

DeepSpeed configs in trainer yamls are bare filenames (e.g. `zero3.json`) resolved
against `src/vlm/config/deepspeed/` automatically. `train.slurm` targets the hyak
`ckpt-all` partition with `--requeue`; training auto-resumes from the last checkpoint
in `trainer.output_dir`.
```

- [ ] **Step 3.8: 审查（按"执行约定"流程，重点核对 get_last_checkpoint 行为与 hydra resolver 语法）→ Commit**

```bash
git add src/vlm/train/training_arguments.py src/vlm/train/train.py src/vlm/config/ train.slurm README.md
git commit -m "fix: parameterize paths via env vars, package-relative deepspeed configs, hyak ckpt-all slurm + auto-resume"
```

______________________________________________________________________

### Task 4: 添加 kernels 依赖（免编译 flash-attn）

**背景:** transformers ≥4.56 内建 kernels-Hub 集成：装了 `kernels` 包后，`attn_implementation="flash_attention_2"` 在本地 flash-attn 缺失时自动 fallback 到 Hub 预编译 kernel，无需源码编译。l40（Ada/SM89）支持 FA2。运行时性能与本地编译版相同。

**Files:**

- Modify: `pyproject.toml`

- Modify: `Makefile:11-23`

- [ ] **Step 4.1: pyproject.toml dependencies 按字母序插入**

```toml
    "kernels>=0.10",
```

- [ ] **Step 4.2: Makefile flash-attn 改为尽力而为**

`install:` 与 `upgrade:` 两处的 `uv pip install flash-attn --no-build-isolation` 前加 `-` 前缀（make 失败继续）并加注释 `# optional: hub kernels are the fallback`。

- [ ] **Step 4.3: 安装并验证**

```bash
uv sync --all-extras --dev
uv run python -c "from kernels import get_kernel; import transformers; print('kernels OK; transformers', transformers.__version__)"
```

- [ ] **Step 4.4: 预热缓存说明（写入 commit message）**

计算节点若无外网：登录节点先 `uv run python -c "from kernels import get_kernel; get_kernel('kernels-community/flash-attn')"` 预热，作业继承 `HF_HOME`。

- [ ] **Step 4.5: 审查 → Commit**

```bash
git add pyproject.toml uv.lock Makefile
git commit -m "build: add kernels dep so FA2 loads from hub when flash-attn is not compiled locally"
```

______________________________________________________________________

### Task 5: 切换到 fused AdamW

**背景:** torch 2.6 下 HF 默认 `adamw_torch`（非 fused）。`VLMTrainer.create_optimizer`（`vlm_trainer.py:68`）走 `get_optimizer_cls_and_kwargs(self.args, ...)`，设置 `optim` 后 fused=True 与自定义差分学习率参数组共同生效。数学等价、零风险；收益为个位数百分比。

**Files:**

- Modify: `src/vlm/config/config_schema.py`（TrainerConfig）

- Modify: `src/vlm/train/training_arguments.py`（get_training_args）

- [ ] **Step 5.1: TrainerConfig 添加字段**（`attn_implementation` 旁）

```python
optim: str = "adamw_torch_fused"
```

- [ ] **Step 5.2: get_training_args 显式传递**（参数是手工逐个传的，漏传则静默失效）

```python
optim = (config.optim,)
```

- [ ] **Step 5.3: 验证**

```bash
CUDA_VISIBLE_DEVICES= uv run python -c "
from vlm.config.config_schema import TrainerConfig
from vlm.train.training_arguments import get_training_args
args = get_training_args(TrainerConfig(name='t', output_dir='/tmp/vlm-test-optim'))
print('optim =', args.optim)
assert 'fused' in str(args.optim)
"
```

- [ ] **Step 5.4: 审查 → Commit**

```bash
git add src/vlm/config/config_schema.py src/vlm/train/training_arguments.py
git commit -m "perf: default to fused AdamW (preserves custom per-component param groups)"
```

______________________________________________________________________

### Task 6: dtype 兼容清理（v5 升级的前置）

**背景:** `torch_dtype=` kwarg 在 4.56 弃用、v5 移除。`vlm.py` 已用 `dtype=`，但 `eval.py:79`、`push_to_hub.py:142,147` 还是旧写法；`eval.py:147` 读 `model.config.torch_dtype`。Task 9 升级前必须先清。

**Files:**

- Modify: `src/vlm/inference/eval.py:79,147`

- Modify: `src/vlm/utils/push_to_hub.py:142,147`

- [ ] **Step 6.1: eval.py 两处修改**

79 行：

```python
dtype = (torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32,)
```

147 行：

```python
images_tensor = images_tensor.to(
    model.device,
    dtype=getattr(model.config, "dtype", None) or next(model.parameters()).dtype,
)
```

- [ ] **Step 6.2: push_to_hub.py 两处修改**

`AutoModel.from_pretrained(...)` 的 `torch_dtype="auto"` → `dtype="auto"`；`AutoProcessor.from_pretrained(...)` 删除 `torch_dtype="auto"` 行（processor 不接受 dtype）。

- [ ] **Step 6.3: 验证**

```bash
uv run python -c "import vlm.inference.eval, vlm.utils.push_to_hub; print('import OK')"
grep -rn "torch_dtype" src/vlm/ && echo "FOUND LEFTOVERS" || echo "clean"
```

- [ ] **Step 6.4: 审查 → Commit**

```bash
git add src/vlm/inference/eval.py src/vlm/utils/push_to_hub.py
git commit -m "chore: torch_dtype -> dtype (v5 forward compat)"
```

______________________________________________________________________

### Task 7: 去掉数据热路径的逐样本 tokenizer deepcopy

**背景:** `preprocess_qwen`（`dataset.py:245-248`）与 `preprocess_llama3`（326-329）每个样本 `copy.deepcopy(tokenizer) + add_tokens`。**不能**直接改共享 tokenizer：`vlm.py` 会在 `len(tokenizer) > vocab_size` 时 resize embedding，deepcopy 正是为了隔离这次 add_tokens（`<image>` 用 -200 哨兵 id 重映射，不需要真实 embedding 行）。正确做法：进程内按 (tokenizer, has_image) 缓存一份副本。

**Files:**

- Modify: `src/vlm/data/dataset.py`

- Test: `tests/test_image_tokenizer_cache.py`

- [ ] **Step 7.1: 写失败测试**

创建 `tests/test_image_tokenizer_cache.py`：

```python
import pytest

transformers = pytest.importorskip("transformers")

from vlm.data.dataset import _get_preprocess_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except OSError:
        pytest.skip("no network / model not cached")


def test_cache_returns_same_object(tokenizer):
    a = _get_preprocess_tokenizer(tokenizer, has_image=True)
    b = _get_preprocess_tokenizer(tokenizer, has_image=True)
    assert a is b


def test_image_variant_has_image_token(tokenizer):
    tok = _get_preprocess_tokenizer(tokenizer, has_image=True)
    assert tok.convert_tokens_to_ids("<image>") is not None
    assert tok.convert_tokens_to_ids("<image>") != tokenizer.unk_token_id


def test_original_tokenizer_not_mutated(tokenizer):
    before = len(tokenizer)
    _get_preprocess_tokenizer(tokenizer, has_image=True)
    assert len(tokenizer) == before


def test_no_image_variant_is_distinct(tokenizer):
    a = _get_preprocess_tokenizer(tokenizer, has_image=True)
    b = _get_preprocess_tokenizer(tokenizer, has_image=False)
    assert a is not b
```

- [ ] **Step 7.2: 运行确认失败**

```bash
uv run pytest tests/test_image_tokenizer_cache.py -v
```

Expected: FAIL，`ImportError: cannot import name '_get_preprocess_tokenizer'`

- [ ] **Step 7.3: 实现缓存函数**（`dataset.py` 在 `preprocess_qwen` 之前，模块级）

```python
# Per-process cache: (id(tokenizer), has_image) -> prepared deepcopy.
# The deepcopy isolation is deliberate — adding '<image>' to the SHARED tokenizer
# would change len(tokenizer) and trigger an unwanted embedding resize in vlm.py.
_PREPROCESS_TOKENIZER_CACHE: dict[
    tuple[int, bool], transformers.PreTrainedTokenizer
] = {}


def _get_preprocess_tokenizer(
    tokenizer: transformers.PreTrainedTokenizer, has_image: bool
) -> transformers.PreTrainedTokenizer:
    key = (id(tokenizer), has_image)
    cached = _PREPROCESS_TOKENIZER_CACHE.get(key)
    if cached is None:
        cached = copy.deepcopy(tokenizer)
        if has_image:
            cached.add_tokens(["<image>"], special_tokens=True)
        _PREPROCESS_TOKENIZER_CACHE[key] = cached
    return cached
```

- [ ] **Step 7.4: 替换两个调用点**

`preprocess_qwen`（245-248 行）与 `preprocess_llama3`（326-329 行）中的：

```python
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
```

均替换为：

```python
tokenizer = _get_preprocess_tokenizer(tokenizer, has_image)
```

保留其后逻辑不动（qwen 的 `tokenizer.chat_template = chat_template` 对缓存副本重复赋同值是幂等的）。

- [ ] **Step 7.5: 运行测试 + 回归**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 7.6: 审查 → Commit**

```bash
git add src/vlm/data/dataset.py tests/test_image_tokenizer_cache.py
git commit -m "perf: cache per-process image-aware tokenizer copy instead of per-sample deepcopy"
```

______________________________________________________________________

### Task 8（可选，防御性）: 视觉塔显式 attn_implementation

**背景:** `_build_vision_model` 不传 attn_implementation，视觉塔静默继承库默认（当前=sdpa，恰好正确）。此改动无性能变化，纯粹钉死行为、防库默认值变化（v5 升级前尤其有意义）。

**Files:**

- Modify: `src/vlm/config/config_schema.py`（VisualEncoderConfig）

- Modify: `src/vlm/models/modeling_vlm.py`（\_build_vision_model）

- [ ] **Step 8.1: VisualEncoderConfig 添加字段**（注意 Task 2 已删除 open_clip 字段后的形态）

```python
@dataclass
class VisualEncoderConfig:
    hf_name: str = MISSING
    output_layer: int | None = None
    use_cls_token: bool = False
    use_all_tokens: bool = False
    attn_implementation: str = "sdpa"
```

（该字段经 `vlm.py` 的 `OmegaConf.to_container(model_cfg.visual_encoder)` 自动流入 VisionConfig，无需改 vlm.py。）

- [ ] **Step 8.2: \_build_vision_model 传入**（基于 Task 2 收敛后的版本）

```python
visual_encoder: PreTrainedModel = AutoModel.from_pretrained(
    vision_config.hf_name,
    trust_remote_code=True,
    attn_implementation=getattr(vision_config, "attn_implementation", None) or "sdpa",
)
```

- [ ] **Step 8.3: 验证 + 审查 → Commit**

```bash
uv run python -c "import vlm.models.modeling_vlm; print('import OK')"
git add src/vlm/config/config_schema.py src/vlm/models/modeling_vlm.py
git commit -m "chore: pin vision tower attn_implementation to sdpa explicitly"
```

______________________________________________________________________

### Task 9: 依赖全面升级（transformers 5.10.1 + torch/accelerate/deepspeed/datasets 最新）与 v5 迁移

**背景:** 用户要求尽量把所有库升到新版，特别是 transformers 5.10.1。当前 pin：transformers 4.56.1 / torch 2.6.0 / accelerate 1.6.0 / deepspeed 0.16.7 / datasets 4.0.0。生态现状（2026-06 调研）：transformers v5.10.1、torch 2.12.0、accelerate 1.13.0、deepspeed 0.19.1、datasets 4.8.5。**这是风险最高的任务**：本仓库用 `type()` 动态建类 + 多处 Trainer 私有 API，每一项迁移点都必须对照 .venv 新版源码逐一验证，不允许凭记忆。

**已知迁移点清单（来自调研，执行时逐项对照 v5.10 实际源码确认）:**

1. `warmup_ratio` 在 v5 弃用、按计划 v5.2 起移除 → `get_training_args` 改用 `warmup_steps`（v5 接受 float\<1 等效 ratio）
1. `torch_dtype` → `dtype`（Task 6 已清理，升级后 grep 复查）
1. `MODEL_MAPPING` / `MODEL_FOR_CAUSAL_LM_MAPPING`（`modeling_vlm.py:43-49` 动态类工厂的根基）在 v5 是否还在原位置
1. `Trainer.get_optimizer_cls_and_kwargs` / `create_optimizer` / `_get_train_sampler(train_dataset)` 签名（`vlm_trainer.py` 覆写）
1. `from transformers.trainer import has_length, is_datasets_available`、`from transformers.trainer_pt_utils import LengthGroupedSampler` 导入路径
1. `prepare_inputs_for_generation` 覆写中的 `inputs.pop("cache_position")`（`modeling_vlm.py:278`）—— v5 缓存/生成接口变化的重点排查对象
1. `low_cpu_mem_usage` kwarg 在 v5 from_pretrained 中的状态
1. 动态 `type()` 类 + 新建 `lm_head` 的 `_init_weights`/`post_init` 行为（v5 自动初始化逻辑变化，防止加载后权重被重新初始化）
1. `VLMProcessor`（`processing_vlm.py`）对照 v5 ProcessorMixin API
1. `report_to=None` 在 v5 的语义（本仓库 4 个 trainer yaml 显式 `report_to: wandb`，但 TrainerConfig 默认 None 需确认不报错）
1. deepspeed 0.19 + 新 torch 的兼容（若 torch 2.12 与 deepspeed 不兼容，逐级回退 2.11/2.10/2.9 找最高兼容版）
1. flash-attn 不重新编译，靠 kernels Hub fallback（Task 4）

**Files:**

- Modify: `pyproject.toml`, `uv.lock`

- Modify: `src/vlm/train/training_arguments.py`（warmup 等 v5 适配）

- Modify: 排查清单中暴露的所有文件

- Create: `devtools/smoke_test.py`

- [ ] **Step 9.1: 升级 pin**

`pyproject.toml` dependencies 改为：

```toml
dependencies = [
    "accelerate>=1.13",
    "blobfile",
    "datasets>=4.8",
    "deepspeed>=0.19",
    "hydra-core",
    "kernels>=0.10",
    "libcst",
    "pillow",
    "protobuf",
    "rich",
    "sentencepiece>=0.2.1",
    "tiktoken>=0.9.0",
    "torch>=2.9",
    "torchvision",
    "transformers==5.10.1",
    "wandb",
]
```

（torch 下限 2.9、不设上限，让 uv 解析 deepspeed/transformers 共同允许的最高版本；若解析结果 \<2.12，记录原因。）

- [ ] **Step 9.2: 重解析并安装**

```bash
uv sync --all-extras --dev
uv run python -c "import transformers, torch, accelerate, deepspeed, datasets; print(transformers.__version__, torch.__version__, accelerate.__version__, deepspeed.__version__, datasets.__version__)"
```

Expected: `5.10.1` + 各库新版本号。若 uv 解析冲突，记录冲突对并逐个放宽。

- [ ] **Step 9.3: 迁移点逐项核查（dispatch 核查 agent，输出核查报告）**

对"已知迁移点清单"的 12 项，逐项在新装的 `.venv/lib/python3.13/site-packages/transformers/`（及 deepspeed）源码中确认现状，必要时用 WebSearch/Context7 查 v5 migration guide。每项输出：仍兼容 / 需修改（给出确切修改）。重点命令示例：

```bash
grep -n "warmup_ratio\|warmup_steps" .venv/lib/python3.13/site-packages/transformers/training_args.py | head
uv run python -c "from transformers import MODEL_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING; print('mappings OK')"
uv run python -c "from transformers.trainer import Trainer, has_length, is_datasets_available; from transformers.trainer_pt_utils import LengthGroupedSampler; print('trainer imports OK')"
uv run python -c "from transformers import TrainingArguments; import inspect; print('warmup_steps' in inspect.signature(TrainingArguments.__init__).parameters)"
```

- [ ] **Step 9.4: 按核查报告逐项修改**

至少包含（以核查结果为准）：

- `training_arguments.py` 的 `get_training_args`：`warmup_ratio=config.warmup_ratio` → 按核查结论改为 `warmup_steps=config.warmup_ratio`（确认 v5 接受 float）或保留（若 5.10 仍兼容 warmup_ratio 则只加 TODO）

- `training_arguments.py` 自定义 TrainingArguments 子类的新增/移除字段比对

- `modeling_vlm.py` 的 `prepare_inputs_for_generation`/`generate` 按 v5 生成接口修正（`cache_position` pop 是否仍需要/仍合法）

- 其余暴露项

- [ ] **Step 9.5: 写冒烟测试（CPU，全链路）**

创建 `devtools/smoke_test.py`：

```python
"""CPU smoke test: build a tiny VLM end-to-end, run forward/backward/optimizer step,
save + reload, verify weights survive round-trip (catches v5 _init_weights regressions).

Run: uv run python devtools/smoke_test.py
Requires network (downloads Qwen2.5-0.5B-Instruct + a small SigLIP) or warm HF cache.
"""

import tempfile

import torch

from vlm.config.config_schema import (
    ConnectorConfig,
    LanguageModelConfig,
    ModelConfig,
    TrainerConfig,
    VisualEncoderConfig,
)
from vlm.vlm import load_model


def main() -> None:
    model_cfg = ModelConfig(
        name="smoke",
        visual_encoder=VisualEncoderConfig(
            hf_name="google/siglip-base-patch16-224",
            output_layer=-2,
            use_all_tokens=True,
        ),
        language_model=LanguageModelConfig(
            hf_name="Qwen/Qwen2.5-0.5B-Instruct",
            max_seq_length=512,
        ),
        connector=ConnectorConfig(name="mlp_2_gelu", type="mlp"),
    )
    trainer_cfg = TrainerConfig(
        name="smoke", bf16=False, fp16=False, attn_implementation="sdpa"
    )
    model, processor = load_model(model_cfg, trainer_cfg)
    model.train()

    # Build a dummy multimodal batch: "<image>\nhi" with one image.
    image_token_index = model.config.image_token_index
    input_ids = torch.tensor([[image_token_index, 1, 2, 3, 4]])
    labels = torch.tensor([[-100, 1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    images = torch.randn(1, 3, 224, 224)

    out = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels, images=images
    )
    loss = out.loss if hasattr(out, "loss") else out[0]
    assert loss is not None and loss.requires_grad, "no differentiable loss"
    loss.backward()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    opt.step()
    print(f"forward/backward/step OK, loss={loss.item():.4f}")

    # Save / reload round-trip: weights must survive (v5 _init_weights regression check).
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        processor.save_pretrained(tmp)
        from vlm.models import get_dynamic_vlm

        VLMForCausalLM, _ = get_dynamic_vlm(tmp)
        reloaded = VLMForCausalLM.from_pretrained(tmp, dtype=torch.float32)
        for (n1, p1), (_, p2) in zip(
            model.named_parameters(), reloaded.named_parameters(), strict=True
        ):
            assert torch.allclose(
                p1.detach(), p2.detach(), atol=1e-6
            ), f"weight mismatch: {n1}"
    print("save/reload round-trip OK")

    # generate() path
    model.eval()
    with torch.no_grad():
        gen = model.generate(inputs=input_ids, images=images, max_new_tokens=4)
    print("generate OK:", gen.shape)


if __name__ == "__main__":
    main()
```

注：`get_dynamic_vlm(tmp)` 要求保存目录的 config 能解析回基座 LLM；若该调用因实现细节失败，改用 `get_dynamic_vlm("Qwen/Qwen2.5-0.5B-Instruct")` 获取类后 `from_pretrained(tmp)`。执行时以实际行为为准修正脚本（属于测试脚手架，允许调整）。

- [ ] **Step 9.6: 运行冒烟 + 全量回归**

```bash
uv run python devtools/smoke_test.py
uv run pytest tests/ -v
uv run python devtools/lint.py
grep -rn "torch_dtype" src/vlm/ && echo "LEFTOVERS" || echo "clean"
```

Expected: 三个 OK + 测试通过 + clean

- [ ] **Step 9.7: （可选，需 GPU）单卡真实冒烟**

```bash
salloc --partition=ckpt-all --account=cse-ckpt --gpus=l40:1 --mem=64G --time=0:30:00 --cpus-per-task=8
# 在分配到的节点上:
cd /mmfs1/gscratch/krishna/leoym/small-vlm && source .venv/bin/activate
VLM_DATA_ROOT=... python -m vlm -cn pretrain-llava trainer.per_device_train_batch_size=2 'trainer.deepspeed=zero2.json' +trainer.max_steps=5
```

（需数据就位；没有数据时跳过，留待数据下载后验证。）

- [ ] **Step 9.8: 深度审查（本任务审查 agent 数量加倍：一个对照 transformers v5 migration guide 逐文件复查，一个跑全部验证命令 + 检查动态类工厂在 v5 下的权重初始化行为）→ Commit**

```bash
git add -A
git commit -m "build!: upgrade to transformers 5.10.1, torch/accelerate/deepspeed/datasets latest; v5 migration"
```

______________________________________________________________________

## 收尾

- [ ] **Final 1: 全量回归**

```bash
uv run pytest tests/ -v
uv run python devtools/lint.py
uv run python devtools/smoke_test.py
```

- [ ] **Final 2: 配置解析烟雾测试**

```bash
for cn in finetune-llava pretrain-llava finetune-qwen pretrain-qwen finetune-llama pretrain-llama; do
  VLM_DATA_ROOT=/tmp VLM_MODEL_ROOT=/tmp uv run python -m vlm -cn $cn --cfg job > /dev/null && echo "$cn OK" || echo "$cn FAIL"
done
```

- [ ] **Final 3: 汇总 commit 历史，交用户确认合并 main**

```bash
git log --oneline main..audit-fixes
```

______________________________________________________________________

## 本计划明确不包含

| 项目                                                                | 原因                                                      |
| ------------------------------------------------------------------- | --------------------------------------------------------- |
| lmms-eval 评测重接                                                  | 用户确认不需要                                            |
| SigLIP 2 编码器                                                     | 用户确认不需要                                            |
| 数据集下载落位（LLaVA-Pretrain / Instruct-150K 到 `VLM_DATA_ROOT`） | 属于运维操作，路径就位后才能跑真实训练                    |
| Liger / cut-cross-entropy 手动集成                                  | `use_liger_kernel` 对动态类是 no-op，需手动集成且收益待测 |
