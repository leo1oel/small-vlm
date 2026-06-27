# NEO pre-Buffer 解剖报告：它到底是不是一个 ViT？

**日期**: 2026-06-05 · **模型**: NEO1_0-2B（Paranioar/NEO1_0-2B-{PT,MT,SFT}，arXiv 2510.14979 "From Pixels to Words"）
**硬件**: 单卡 A100 80GB / L40s · **代码**: `small-vlm/neo_analysis/` · **数据**: 本目录 `data/` · **图**: `figures/`

---

## TL;DR

> **问题**：NEO 在 LLM 前面加了 12 层随机初始化的 pre-Buffer，第一阶段冻结 LLM 只训 pre-Buffer。这样训出来的 pre-Buffer 是不是只是个（生成式损失训练的）ViT？里面有没有 vision-text 交互？如果没有，是不是"和分开训没区别"？

> **答案**：**pre-Buffer 内不存在任何功能性的跨模态交互——对 PT 成立，对最终 SFT 模型同样成立。**
> 把最终模型重构成"两条单模态 pre-Buffer 通路（共享权重）+ 28 层 post-LLM 融合"，在 POPE / MME 真实基准上**零精度损失**（0.907/0.871 vs 0.910/0.875，Fig 9）。所有跨模态融合发生在 post-LLM 的 L16–23（恰好对应原 Qwen3 的第 4–11 层，与 LLaVA 类模型的融合层位一致）。
> 唯一的非平凡耦合是**接口性质**的：① vision token 把 ~23% 注意力质量投给前缀模板 token（attention sink），剥夺 sink 会崩溃；② post-LLM 期望 pre-Buffer 输出的残差流尺度，喂原始 embedding 会崩溃。这两者都不是"融合"。
> 所以：**NEO-2B 在功能上 = 一个 12 层生成式 ViT（带 sink）+ 一个单模态文本前处理器（共享同一套权重）+ 一个 LLaVA 式的融合 LLM。** "native/unified" 与"模块化"的区别只存在于参数共享和训练信号，不存在于推理时的信息流。

---

## 1. 背景与架构事实（源码核实）

NEO（EvolvingLMMs-Lab）声称做 encoder-free 的 native VLM：无独立视觉 encoder，patch 直接进 LLM 主干。代码核实的关键事实：

1. **`NEOVisionModel` 不是 ViT**：仅 Conv2d patch embed (16×16) + GELU + 2D-RoPE + Conv2d 2×2 下采样，**零层 attention**（`modeling_neo_vit.py`）。全部视觉计算在 LLM 主干内。
2. **pre-Buffer = 12 层随机初始化的 Qwen3DecoderLayer**，插在 28 层预训练 Qwen3 之前（2B 配置；checkpoint 权重平移映射 `modeling_neo_qwen3.py:899-907`）。三路 RoPE（t/h/w）。
3. **Attention mask 全网统一**：`(因果 ∧ 同文档) ∨ (同图内双向)`（`modeling_neo_chat.py:80-123`）。整张图共享一个 temporal index。**架构上不禁止任何跨模态交互**——是否交互完全由学习决定。
4. **论文训练方案**：PT 阶段只训 patch embed + pre-Buffer + QK 的 H/W 投影和 norm（论文原文），190k 步、345M 图文对、lr 8e-4；MT/SFT 全解冻（"the partition dissolves"）。
   ⚠️ **开源训练代码的冻结逻辑是死代码**：`train.py:28-48` 的 `set_model` 从未被调用，且按层号选参的分支有解析 bug（`parts[2]=='layers'` 永远非数字）。开源代码 ≠ 论文方案，复现需自行实现冻结。
5. **论文自己没做任何 pre-Buffer 的机制分析**（无 attention 分析/probing/CKA/knockout）；最接近的是 PB3/Fig6（pre-Buffer 拆出重训当 encoder，与 CLIP/SigLIP 差 1.7–3.7%）。本报告的 H1/H2/H4 分析均为新结果。

### 假设分解

| | 命题 | 结论 |
|---|---|---|
| **H1** | pre-Buffer 中 vision 表征不被 text 条件化（text→vision ≈ 0） | ✅ 成立（全阶段） |
| **H2** | text 在 pre-Buffer 内不读取视觉信息（vision→text ≈ 0，读取推迟到 post-LLM） | ✅ 成立（全阶段；QA 任务零损失，原生 caption 任务 PT 仅 +0.43 nats、MT/SFT 切断后反而**改善** −0.5~−0.8） |
| **H3** | pre-Buffer 输出 ≈ 通用 ViT 特征 | ✅ 成立（CKA 0.76–0.80 vs CLIP/SigLIP/DINOv2） |
| **H4** | pre-Buffer 对 text ≈ 恒等映射 | ◐ 修正后成立：不是逐层恒等，而是"**单模态**"——text 通路做真实的文本计算（MT/SFT 后不可旁路），但完全不依赖视觉输入 |

---

## 2. 实验设置

- **Checkpoints**: PT / MT / SFT 三阶段全开源（各 ~5.8GB；PT 的 config 误标 `internvl_chat` 且 auto_map 指向不存在的文件，需修复加载，见 `neo_analysis/neo_local.py`）。
- **环境**: 专用 venv（torch 2.5.1 + transformers 4.57.1，按 NEO 官方 pin）。⚠️ flex_attention 在 L40s (sm89) 上编译失败（有效 head_dim=256 超 shared memory），全部实验用数学等价的 **SDPA + 稠密 mask** 替换（`patch_sdpa_attention()`，与 builtin forward 数值逐位一致）。
- **数据**: POPE（均衡采样 300，3 类各 100）、MME 感知子集（existence/count/position/color/OCR，280 题）、COCO val2017 150 图 + captions、自建 15 题受控 QA（合成图形/数数/OCR + 人工核验的 COCO 图）。
- **评测方式**: yes/no 基准用单次 forward 的 `logit("Yes") − logit("No")`；NLL 实验用 teacher-forced 答案 NLL。手术实验全部先断言 manual trunk == builtin forward。
- ⚠️ 第一轮 POPE 用了偶数步长采样，与数据 yes/no 交替排列混叠成全 yes 子集——已改随机采样并加均衡断言；本报告 POPE 数字均来自均衡版本。

### 行为基线（冒烟测试，13 题受控 VQA）

| | SFT | MT | PT |
|---|---|---|---|
| 正确率 | 13/13 | 12/13 | 不适用 |
| 行为 | 正常 QA | 正常 QA（且能直接说出 STOP 牌倒置） | **完全无视问题，输出 alt-text 风格 caption**（含网页爬取残留、复读），但视觉 grounding 准确 |

PT 是一个图像条件 captioner——这本身就预示了它的 pre-Buffer 是"视觉编码器"而非"融合器"。

---

## 3. 结果

### 3.1 逐层更新画像 + Qwen 恒等下限（Fig 1）

每层残差相对更新量 `‖Δh‖/‖h‖`，按 text/vision 位置分开统计；对照 vanilla Qwen3-1.7B-Base 同指标。

- **pre-Buffer L01–07 对 text 的更新只有 0.02–0.09（cos≈0.999）**，vanilla Qwen3 同位层是 0.36–2.5——**低 5–20 倍**。"残差网络本来就近似恒等"的通用解释被基线排除：这是**学出来的 text 跳过**。
- 镜像不对称：pre-Buffer 重度加工 vision（0.1–1.4）、几乎不动 text；post-LLM 第一层（L12）猛烈加工 text（0.36–0.54）、几乎不动 vision（0.03–0.06）。
- 结构签名：L11 有 1.6–1.7 的"交接变换"；NEO L14 与 vanilla Qwen3 L02（各自最大的改写层）一一对应——层平移映射肉眼可见。
- 三个 stage 的画像几乎重合：解冻没有改变这个分工结构。

### 3.2 边界手术：naive 解耦 vs 正确解耦（Fig 2, 4, 9 — 核心结果）

在 layer-12 边界做三种"naive"手术（B_txt：text 行替换为原始 embedding；B_vis：vision 行替换为去文本重算；B_full：两者同时 = naive LLaVA 重构），以及"正确"的 split 手术（split_txt/split_vis/split_full：**单模态 pre-Buffer 通路**，vision 通路保留前缀 sink，text 通路完整因果文本处理）。

**真实基准准确率（均衡 POPE n≈300 / MME 感知 n=280）：**

| SFT | intact | naive B_txt | naive B_vis | naive B_full | **split_txt** | **split_vis** | **split_full** |
|---|---|---|---|---|---|---|---|
| MME | 0.875 | 0.482 | 0.521 | 0.496 | 0.871 | 0.871 | **0.871** |
| POPE | 0.910 | 0.607 | 0.470 | 0.560 | 0.907 | 0.910 | **0.907** |

（MT 同模式：MME split_full = intact = 0.836 逐位相同，POPE naive 崩到 0.47–0.61。naive 手术带强系统性偏置：B_full 预测 yes 率 0.97、B_vis 0.20。）

**解读**：naive 解耦崩溃、正确解耦无损——两者的差异精确隔离出 pre-Buffer 的全部"耦合"是两个**接口伪影**：
1. **sink 剥夺**（B_vis 砍掉了 vision 通路里的前缀模板 token）；
2. **残差流尺度失配**（B_txt 给 post-LLM 喂了它从没见过的原始 embedding 深度的 text 行）。
两者都与跨模态信息无关。**最终模型功能上 = 两条单模态通路 + post-LLM 融合。**

辅助证据（tier-1 NLL，15 题受控 QA + 150 COCO caption）：naive B_txt 的 ΔNLL 从 PT +0.10 → MT +2.74 → SFT +2.81；text-only 对照（+0.47 vs 带图 +2.81）一度提示"图像依赖的融合"，**后被 knockout 和 split 推翻**——图像依赖来自尺度失配在多模态序列里的级联放大，不是 pre-Buffer 内的融合（Fig 2 右图标题已按最终解释修正）。

**格式混淆的修正（caption-chat 实验）**：tier-1 中 PT 的 B_txt≈+0.10 是"QA 格式下 PT 不使用文本"的混淆。在 PT 原生分布（chat 模板 + caption 作为 assistant turn，base NLL 3.35 vs 裸拼接格式 13.2）上，naive B_txt 三个 stage 一致地大（PT +3.74 / MT +3.34 / SFT +3.14）——naive raw-bypass 在原生任务上对**所有**阶段都是大代价；它是不是伪影由 caption-knockout（§3.3 末）判定——**判定结果：~88% 是伪影**。

### 3.3 因果定位：跨模态读取发生在哪里（Fig 6 — knockout）

按 4 层带切断 text-query→vision-key 注意力边（均衡 POPE n=102）：

| 切断范围 | SFT acc | MT acc |
|---|---|---|
| intact | 0.912 | 0.922 |
| **pre-Buffer 全段 [0–11]** | **0.922（无损）** | **0.931（无损）** |
| post-LLM 全段 [12–39] | 0.422（崩溃） | 0.382（崩溃） |
| 细带峰值 | **L16–19 (−0.069) / L20–23 (−0.108)** | L16–19 (−0.108) / L20–23 (−0.098) |
| vision→prefix sink [0–11] | 0.373（崩溃） | 0.480（崩溃） |

- **text 读图 100% 发生在 post-LLM**，峰值 L16–23 = 原 Qwen3 的第 4–11 层——与 Basu et al.（arXiv 2411.18620）在 LLaVA-1.5 上定位的"低中层搬运视觉信息"完全一致。**NEO 和 LLaVA 用同一套层位做融合。**
- 单带 drop 小（≤0.11）但全段切断崩溃 → post-LLM 内的读取是冗余分布的。
- sink 切断崩溃，确认 B_vis 损失的机制。

**原生任务闭环（caption-knockout，Fig 10）**——对 PT 也成立的最终裁决。chat 格式 caption（PT 原生分布）NLL 下，两种独立手术（切断 pre-Buffer 内 text→vision 边 vs 单模态 text 通路）收敛到同一数值：

| ΔNLL (nats) | t2v_pre 切断 | split_txt 单模态 | t2v_post 切断 | sink 切断 |
|---|---|---|---|---|
| PT | +0.43 | +0.43 | +6.03 | +6.59 |
| MT | **−0.49** | **−0.49** | +1.06 | +1.06 |
| SFT | **−0.78** | **−0.78** | +0.93 | +0.85 |

- PT 在原生任务上 pre-Buffer 内的真实跨模态读取只值 +0.43 nats（post-LLM 的 7%）；naive B_txt 的 +3.74 有 ~88% 是尺度伪影。
- **MT/SFT 阶段切断 pre-Buffer 跨模态注意力反而改善 NLL**（−0.5 ~ −0.8）——那些注意力边是净噪声。
- 两种手术逐 stage 收敛到同一数值（+0.43/+0.43、−0.49/−0.49、−0.78/−0.78），方法学自洽。

### 3.4 注意力质量流（Fig 7）

30 个 POPE 样本上逐层统计类间注意力质量（prefix/vision/question/last）：

- **vision→prefix（sink）**：pre-Buffer 内均值 0.23–0.24，在 L11 边界附近峰值 0.43——占 key 总数 <1% 的几个模板 token 吃掉近四分之一的注意力质量。**三个 stage 完全一致：sink 在 PT（冻结期）形成并永久保留。**
- **question→vision**：pre-Buffer 内 ~0.05，跨过 L12 瞬间跳到 0.5–0.6（10 倍）。last→vision 在 L12 处尖峰 0.55。
- 注意：L12–15 的注意力质量很高但 knockout 显示功能上可有可无（切断零损失）——**质量 ≠ 必要性**，L12 的大规模读取可能是冗余的"全图 gist 拷贝"，决策关键的读取在 L16–23。

### 3.5 表征相似度：pre-Buffer 学到的是通用视觉特征（Fig 8 — CKA）

150 张 COCO 图上，NEO 各层 vision token mean-pool 特征与冻结 encoder 各层做 linear CKA（取 max over encoder layers）：

| | vs CLIP-L-336 | vs SigLIP-base | vs DINOv2-base |
|---|---|---|---|
| PT @L11（pre-Buffer 出口） | 0.757 | 0.800 | 0.792 |
| SFT @L11 | 0.643 | 0.641 | 0.654 |
| SFT 全层最佳 | 0.812 | 0.867 | 0.851 |

- **纯 next-token 生成损失、无任何对比学习，pre-Buffer 出口特征与三大视觉 encoder 的 CKA 达 0.76–0.80**——H3 成立，与论文 PB3 复用实验互证。
- 解冻后（MT/SFT）出口处 CKA 降到 ~0.65，但曲线形态显示"最像 encoder"的点移到 pre-Buffer 中部（L8–10，0.78），post-LLM 前段降到 0.5（任务化改写），深层（L32–38）回升到 0.85。

### 3.6 text→vision 方向：表征不变性（Fig 3）

同图配等长不同前置问题（严格控制 RoPE t-index），测 vision 态逐层漂移：

- pre-Buffer 出口（L11）漂移：PT 0.122 → MT 0.052 → **SFT 0.045**（不同图天花板 1.35，即 ~3%）。**训练方向是越来越不变**——PT 的较高漂移是没见过 text-first 布局的 OOD 噪声，不是习得条件化。
- img-first 布局漂移恰为 0.00000（结构必然，harness 自检）。
- 功能验证：B_vis(text_first) − B_vis(img_first) ≈ 0 ——把问题从 vision 视野中移除无任何 NLL 代价。

### 3.7 外部对照：Gemma-4-12B-it（端到端训练的 native VLM，Fig 11）

为判断 NEO 的模态分区是普遍规律还是其训练课程的产物，在 google/gemma-4-12B-it（`gemma4_unified`，2026-06）上复刻了同套实验。Gemma-4 是更彻底的 native：视觉塔为**纯线性** patch embedder（LN→Dense→LN + 2D 位置嵌入，零 attention），48 层统一主干无任何 pre-buffer 分段，图像块内双向 attention（与 NEO 同语义），端到端联合训练（含原生 audio）。

| 指标 | NEO-2B SFT | Gemma-4-12B |
|---|---|---|
| 早期层 text 相对更新量 | 0.02–0.09（学会跳过） | **1.0–1.8（重度加工）** |
| question→vision 注意力质量（早期层） | ~0.05 | **0.22（从 L0 开始）** |
| text→vision 条件化（漂移/天花板） | 全程 ~3% | 随深度增长至 **~46%** |
| 切断早期层 text→vision 的精度损失 | 0（pre-Buffer 全段） | **0（L0–11 全段！）** |
| 决策关键的读取层带 | L16–23（相对深度 0.4–0.6） | L16–31（相对深度 0.33–0.65） |
| 全段切断（中后段）后 | 0.42（崩溃） | 0.32（崩溃） |

**双重结论**：
1. **表征层面，NEO 的模态分区是课程产物**：Gemma-4 从 L0 就有大规模双向交互（注意力质量 22%、双模态同等加工、深层 vision 态被 text 强条件化）——端到端训练的 native 模型不会自发产生 NEO 那种"前段单模态"结构；
2. **功能层面，两者惊人一致**：尽管 Gemma-4 早期就交互，切断其前 12 层的 text→vision 边照样**零精度损失**——早期交互是冗余的。决策关键的跨模态读取在两个模型（以及文献中的 LLaVA）里都住在**主干中段**（相对深度 ~0.35–0.65）。"功能性融合居于中段"看起来是跨架构、跨训练方式的普遍规律；架构与课程只决定早期层是"冗余地混合"还是"严格地分工"。

**可解耦深度曲线（Fig 12）**——把"能否像 NEO 一样解耦"做成扫描：在层 [0..d] 内完全隔离两模态（mask 等价于 split 手术；模板与 `<start_of_image>` 锚点保留给 vision 当 sink），扫描边界 d：

| d（占 48 层） | 4 | 8 | 12 | 16 | 20 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|---|---|
| img_first（intact 0.824） | 0.824 | 0.794 | **0.824** | 0.765 | 0.667 | 0.588 | 0.314 | 0.314 |
| text_first（intact 0.745） | 0.745 | 0.745 | 0.755 | **0.745** | 0.725 | 0.696 | 0.373 | 0.343 |

- **Gemma-4 同样可以无损解耦到主干前 ~1/4–1/3**（img_first 至 L12、text_first 甚至至 L16），与 NEO 的 split@L12/40 落在同一相对深度的零线上；
- text_first 下 vision 真实的问题条件化（深层漂移达 46%）被隔离移除，依然零损失——**连"真实存在"的早期/深层 text→vision 条件化对决策也是冗余的**；
- 悬崖在相对深度 ~0.4–0.65，精确对应功能融合区；
- 附带发现：第一版手术误切 `<start_of_image>` 锚点导致 d4 即崩（0.60）——**vision-token 对图像前锚点 token 的 sink 依赖在 Gemma-4 上同样存在**，与 NEO 的 `<img>`/前缀 sink 现象互证，是跨模型的普遍机制。

### 3.8 Vision-centric 基准泛化：MMStar + MMVP（Fig 13）

针对"yes/no 任务太简单"的质疑，在两个 vision-centric MCQ 基准上复测（单 forward 字母 logit 打分，与 lmms-eval 的 letter-matching 约定一致）：**MMStar**（1500 题，人工筛选 vision-indispensable + 无泄漏，6 大能力类别）、**MMVP**（300 题，CLIP-blind 图像对，专测细粒度视觉属性）。

**NEO（split 解耦全条件）：**

| | intact | split_txt | split_vis | **split_full** |
|---|---|---|---|---|
| SFT MMStar | 0.514 | 0.518 | 0.519 | **0.516** |
| SFT MMVP | 0.663 | 0.650 | 0.663 | **0.660** |
| MT MMStar | 0.486 | 0.484 | 0.487 | **0.484** |
| MT MMVP | 0.647 | 0.643 | 0.643 | **0.643** |

MMStar 六大类别（含 fine-grained perception −0.8pt、logical reasoning −1.2pt）全部 ±2pt 内——**没有任何能力类别为解耦买单**。

**Gemma-4（隔离深度扫描）：**

| d / 48 层 | 0 | 8 | 12 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|
| MMStar | 0.543 | 0.543 | **0.541** | 0.525 | 0.375 | 0.269（随机） |
| MMVP | 0.827 | 0.840 | 0.827 | **0.830** | 0.657 | 0.500（随机） |

- **MMVP（最依赖细粒度视觉的基准）完全隔离到 L16（主干 1/3）零损失**（0.830 vs 0.827）；MMStar 免费窗口到 L12（−0.2pt）；
- 悬崖位置（d=24 起）与 POPE 扫描和功能融合区（L16–31）完全一致；
- d=48（全程隔离 = 永不融合）精确退化到随机线（0.269≈1/4、0.500=1/2）——任务确实 vision-indispensable，打分方式无泄漏，对照自洽。

**结论**：早期跨模态交互的功能冗余性不是简单任务的伪象——在专门设计来刁难细粒度视觉感知的基准上同样成立。

---

### 3.9 统一全层融合指标：跨模型不变量（Fig 14, 15）

前面的实验用"采样几个深度"定位融合区。为得到一个**可引用的单一指标**并做跨架构比较，定义三条**全层**曲线（每层都测），统一在 VMCBench dev 上（与小模型 SFT 的 lmms-eval 同基准、同字母 logit 打分），详见 `FUSION_METRICS_EXPLAINED.md`：

- **`cost(d)` / `retained(d)`（功能轴）**：累积屏蔽层 `[0..d)` 的全部 text→vision 注意力后的准确率；`retained(d) = acc(真图) − acc(换错图)` = 屏蔽后仍存活的"可用图像信号"。零假设带 = 换图条件。**功能融合质心 CoM** = 按 `retained` 逐层下降量加权的平均层（"融合中心在第几层"的单一标量）。
- **`rho(l)`（表征轴）**：`‖h(I)−h(I')‖ / ‖h(I)−h(∅)‖` = 答案位隐状态对"是哪张图"的敏感度 / 对"有没有图"的总敏感度 = **内容融合率**。

三个 native/unified VLM 对照——28/40/48 层、小 SFT / 原生 pre-Buffer / 端到端原生、强弱差一倍：

| 模型 | 层数 | intact | swap | R0(图像依赖) | 功能 CoM(绝对) | **功能 CoM(相对深度)** |
|---|---|---|---|---|---|---|
| small-vlm bee-mix (ckpt-5000) | 28 | 0.396 | 0.374 | +0.022 | L15.6 | **0.56** |
| NEO1.0-2B (SFT) | 40 | 0.683 | 0.443 | +0.240 | L21.8 | **0.54** |
| NEO1.0-9B (SFT) | 42 | 0.763 | 0.557 | +0.207 | L21.9 | **0.52** |
| Gemma-4-12B-it | 48 | 0.677 | 0.443 | +0.233 | L23.2 | **0.48** |

**核心结论：功能融合质心是跨架构不变量，落在相对深度 0.48–0.56（约主干一半深处），与层数（28/40/42/48）、规模（2B/9B/12B）、架构、训练范式无关。**

> 零假设稳健性（swap robustness，`results_swap_robustness.json`）：bee-mix-5000 上用 3 张不同错图，R0 = +0.033 / +0.030 / +0.030（SD≈0.0015）—— retained/R0 对错图的具体选择不敏感，null 稳健。 三模型的 `retained(d)/R0` 曲线都**在前 ~30–40% 深度保持满信号**（解耦无损），然后在相对深度 0.4–0.7 被抹掉：

- **NEO 满信号精确撑到相对深度 0.30——正好是 pre-Buffer 边界（L12/40）**：屏蔽整个 pre-Buffer 的 text→vision 完全免费 → 从这套定量指标看，**pre-Buffer 精确落在融合区之前，确实是"融合前的 ViT"**（§3.2/3.3 的独立佐证）；
- Gemma-4 相对深度 0.3 前满信号，0.4–0.6 塌掉；
- bee-mix 因 R0 太小（0.036，该模型本就几乎不依赖图像）曲线最噪，但同样在 0.3–0.5 塌。

**表征轴 `rho(l)` 的形态按架构各异**：bee-mix 高起衰减（L1=0.79→0.25）、Gemma 低起爬升（0.31→0.84）、NEO 全程高（0.75→1.1）。即"图像内容**何时进入**文本表征"是架构/课程的产物，但"**何时被用来出答案**"（功能融合）收敛到相对深度 ~0.5。这是 §3 反复出现的 **①信息在不在 ≠ ②信息有没有被用上** 的跨模型量化证据。

> 附带观察：bee-mix 的 R0 仅 +0.036（intact 0.41）说明该规模模型在 VMCBench 上大部分准确率来自文本先验；NEO/Gemma 的 R0≈+0.24（intact 0.68）才是真正依赖图像的强 VLM。融合**位置**不变，但融合**幅度**随模型能力差一个量级。

#### 3.9.1 融合位置随训练阶段的演化（NEO PT→MT→SFT，Fig 15）

把同一套指标作用于 NEO 的三个训练阶段（开源 checkpoint `Paranioar/NEO1_0-2B-{PT,MT,SFT}`）：

| 阶段 | intact | swap | R0 | 功能 CoM(相对) | rho@L1 | rho@pre-Buffer边界(rel 0.30) |
|---|---|---|---|---|---|---|
| PT | 0.267 | 0.287 | **−0.020** | （不可读） | 0.49 | 1.11 |
| MT | 0.650 | 0.430 | +0.220 | 0.60 | 0.58 | 1.07 |
| SFT | 0.683 | 0.443 | +0.240 | 0.54 | 0.67 | 1.10 |

两条结论：

1. **功能融合位置在训练后期稳定，不随阶段前移**。PT 是纯 caption 模型（intact 0.27≈随机线、R0=−0.02：它无视 MCQ 问题、答案位不依赖图像），其**准确率口径的功能指标退化为噪声、不可读**——这一段以表征指标 rho 为准。可测量的 MT/SFT 两阶段，功能融合质心都落在相对深度 0.54–0.60，且两阶段屏蔽整个 pre-Buffer（rel 0.30）的 text→vision 仍**零损失**（retained/R0 ≈ 1.0）。**训练让模型答得更准（intact 0.65→0.68），但没有把功能融合搬进 pre-Buffer。**

2. **表征内容融合随训练增强，且在每个阶段都于 pre-Buffer 出口达峰**。rho 从 L1 上升、在相对深度 0.30（pre-Buffer 边界）见峰 ~1.1（三阶段一致），经 LLM 交接处回落后再起。rho@L1 随训练单调上升（PT 0.49 → MT 0.58 → SFT 0.67）：**文本流对图像内容的敏感度随训练变强、变早**。

   与 §3.6 的方向对照构成一个干净的非对称图景：训练同时让 **vision 更不依赖 text**（prompt-swap 漂移 0.122→0.045，§3.6，视觉侧愈发像纯 ViT）和 **text 更依赖图像内容**（rho@L1 0.49→0.67，文本侧 grounding 增强）。NEO 的训练学的是"把视觉编码器养干净 + 把图像内容读进文本流"，而**不是**把两者在 pre-Buffer 里混起来——这与全报告的主结论一致。

#### 3.9.2 跨版本与跨尺度：NEO1.0/1.5 × 2B/9B（Fig 16）

把同一套指标作用于 NEO 的四个变体（`Paranioar/NEO1_{0,5}-{2B,9B}-SFT`），2B 是 12 层 pre-Buffer（rel 0.30），9B 是 6 层（rel 0.143）：

| 变体 | pre-Buffer | intact | R0 | 功能 CoM(相对) | rho@L1 |
|---|---|---|---|---|---|
| NEO1.0-2B | 12 (rel 0.30) | 0.683 | +0.240 | 0.54 | 0.64 |
| NEO1.0-9B | 6 (rel 0.143) | 0.763 | +0.207 | 0.52 | 0.52 |
| NEO1.5-2B | 12 (rel 0.30) | 0.757 | +0.273 | 0.53 | 0.86 |
| NEO1.5-9B | 6 (rel 0.143) | 0.807 | +0.240 | 0.51 | 0.64 |

三条结论：

1. **功能融合位置对版本和尺度都不变**：1.0/1.5 × 2B/9B 全部 funcCoM 相对深度 0.51–0.54。连同 §3.9 的 bee-mix/Gemma，**八个模型配置（28–48 层、2B–12B）的功能融合质心全部落在相对深度 0.48–0.56**——这是目前最强的跨架构不变量证据。

2. **pre-Buffer 在两个尺度都落在融合区下方，且更大的 LLM 用更薄的 pre-Buffer**：2B 的 pre-Buffer 在 rel 0.30，9B 只在 rel 0.143，但两者的功能融合都在 ~0.52。这说明**更大的 LLM 自身吸收了更多视觉编码**，只需更薄的专用 pre-Buffer——pre-Buffer 是"视觉编码容量"而非"融合定位器"（与 §3.9 把它读作内生视觉编码器一致）。

3. **训练只动融合的"幅度"，不动"位置"**：NEO1.5 比 1.0 的 R0（2B：0.27 vs 0.24）和 rho@L1（2B：0.86 vs 0.64）都更高，准确率也更高（intact 0.76 vs 0.68）——更好的训练让模型**更依赖图、图像内容更早进入文本流**——但 funcCoM 相对深度不动。**融合幅度可训，融合位置是结构常量。**

---

## 4. 结论

1. **"pre-Buffer 是不是用生成式预训练训出来的 ViT？"——是，而且比预想的更彻底。** 它是一个带 attention-sink、与 CLIP/SigLIP/DINOv2 表征高度对齐（CKA 0.8）的 12 层双向视觉 encoder，外加一条与之共享权重、互不读取的单模态文本通路。
2. **"训完之后有没有交互？"——没有功能性交互。** text→vision：表征漂移 3%、移除零代价；vision→text：pre-Buffer 全段 knockout 零损失。全部融合在 post-LLM L16–23（≘ Qwen3 L4–11，与 LLaVA 同位）。
3. **"没有交互是不是和分开没区别？"——推理时：正确解耦后无区别（split_full = intact）。** 区别只在：(a) 参数共享（同一套权重跑两条单模态通路，省一个独立 encoder 的参数）；(b) 训练信号（encoder 由穿过 LLM 的生成式梯度端到端训练，这可能正是 CKA 高、PB3 接近 CLIP 的原因）；(c) 两个接口约定（sink anchor + 残差流尺度），它们让 naive 拆解失败，但本质是工程接口而非信息融合。
4. **对论文叙事的修正**："deep pixel-word integration within the pre-Buffer" 不被数据支持——integration 在 post-LLM；"the partition dissolves" 只对 text 通路的不可 raw-bypass 性成立，**模态分区在功能上从未溶解，反而随训练更固化**（漂移 0.122→0.045）。NEO 真正与 LLaVA 不同的是封装方式，不是计算图。
5. **跨架构不变量（§3.9）**：用统一的全层指标（功能 CoM 相对深度）量化，**功能融合质心稳定在相对深度 0.48–0.56**，跨 8 个模型配置（bee-mix-2B / NEO1.0-2B,9B / NEO1.5-2B,9B / Gemma-4-12B；28–48 层、2B–12B、三种范式）全部成立。pre-Buffer 边界（2B rel 0.30、9B rel 0.143）在两个尺度都落在融合区之前——"pre-Buffer 是融合前的 ViT"从定性结论升级为可测的位置关系；且**更大的 LLM 用更薄的 pre-Buffer**（9B 仅 6 层）印证 pre-Buffer 是视觉编码容量而非融合定位器。训练（1.0→1.5）只增大融合幅度（R0、rho），不移动融合位置。

### 工程含义

- pre-Buffer + patch embed 可以**无训练**拆出来当视觉 encoder 用（保留 sink 前缀），输出在 L8–11 取最优。
- 推理优化：两条单模态通路可以并行/缓存（图像特征可离线预计算——和 LLaVA 一样），split 重构零精度损失。
- 若想真正把融合提前到 pre-Buffer，需要显式机制（如 text-first 数据占比、跨模态辅助损失）——单纯共享权重 + NTP 不会自发产生早期融合。

## 5. 局限与未尽事项

- 评测以 yes/no 基准（POPE/MME 感知）+ NLL 为主；开放生成任务（caption/OCR 长答案）下 split 等价性只验证了 NLL 量级，未跑完整生成评测。
- knockout/attn-mass 样本量 102/30，定位结论的带宽（L16–23）有 ±1 带的不确定性。
- CKA 用 mean-pool 图级特征；patch 级对应关系未测（涉及 grid 不匹配）。
- 只分析了 2B；9B（6 层 pre-Buffer）未验证。
- PT 的 yes/no、MCQ 与 chat 格式实验不可解读（caption 模型，§3.9.1 中 PT 的 R0=−0.02 再次证实）；PT 专属结论只基于表征指标（rho、CKA、漂移）。功能融合位置的阶段演化只有 MT/SFT 可测。
- 多图/视频/text-first 训练数据下 pre-Buffer 是否仍单模态，未测（NEO1_5 可能不同）。

## 6. 文件索引

| 文件 | 内容 |
|---|---|
| `figures/fig1_update_norms.png` | 逐层更新画像 + Qwen 基线 |
| `figures/fig2_surgery.png` | tier-1 naive 手术 ΔNLL + text-only 对照 |
| `figures/fig3_prompt_swap.png` | text→vision 表征漂移 |
| `figures/fig4_real_vqa.png` | naive 手术的真实基准精度 |
| `figures/fig5_caption.png` | caption 格式手术 |
| `figures/fig6_knockout.png` | 逐层带 knockout 定位 |
| `figures/fig7_attn_mass.png` | 类间注意力质量流 |
| `figures/fig8_cka.png` | CKA vs CLIP/SigLIP/DINOv2 |
| `figures/fig9_split_headline.png` | **头图：正确解耦零损失** |
| `figures/fig10_caption_knockout.png` | 原生任务闭环：pre-Buffer 跨模态注意力 ≈ 可白拿掉 |
| `figures/fig11_gemma4_compare.png` | NEO vs Gemma-4：分区是课程产物，功能融合都在中段 |
| `figures/fig12_decouple_sweep.png` | 可解耦深度曲线：两个 native VLM 都可无损解耦前 ~1/3 |
| `figures/fig13_vision_bench.png` | MMStar/MMVP：vision-centric 基准上结论同样成立 |
| `figures/fig14_fusion_crossmodel.png` | **统一全层指标：功能融合质心跨架构不变（相对深度 ~0.5）** |
| `figures/fig15_fusion_stages.png` | **NEO PT→MT→SFT：融合位置随训练阶段的演化** |
| `figures/fig16_neo_variants.png` | **NEO 1.0/1.5 × 2B/9B：融合位置对版本和尺度不变，pre-Buffer 随尺度变薄** |
| `figures/fig17_fusion_density.png` | **f(d) 功能融合密度 + FDI（早/晚融合标量），六模型全为 mid fusion (0.48–0.56)** |
| `figures/fig18_attn_vs_sail.png` | **image 注意力质量 vs SAIL：我们的 native 模型落在"模块化"区间(10–30%)，sink 吸走质量** |
| `FUSION_METRICS_EXPLAINED.md` | 三个融合指标（cost/retained、rho、nu）的定义、直觉与读法 |
| `data/results_*.json` | 全部原始结果（含 `results_gemma4.json`） |
| `../neo_analysis/results_fusion_full_{base2000,gemma4,neo,neo_PT,neo_MT}.jsonl` | §3.9 全层指标原始数据（zero-skip） |
| `../neo_analysis/` | 全部实验代码（`neo_local.py` 为加载器，`tier1_experiments.py`/`exp_*.py` 为各实验；`neo_fusion_full.py`/`gemma4_fusion_full.py` 为统一指标探针） |
| `../devtools/{aux_fusion_full,fusion_crossmodel_figure,neo_stage_figure}.py` | bee-mix 探针 + 跨模型/跨阶段出图 |

*实验均可复现：`source /mmfs1/gscratch/krishna/leoym/envs/neo/bin/activate && python exp_<name>.py`*
