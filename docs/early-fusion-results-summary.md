# Early-fusion / 视觉处理增强实验 结果汇总

实验对象:encoder-free 原生小 VLM(Qwen3-1.7B + raw 48px patch + 纯 projection connector),
SFT-only(~1M caption 对 / 657M tokens,无视觉预训练阶段)。
对照基线:`sft-unified-bee-mix`(baseline)。所有评测 lmms-eval,单 seed,终点 checkpoint-5000。
VMCBench(dev,多选 A/B/C/D)/ MMVP / POPE。

## 终点对照(checkpoint-5000,单 seed)

baseline-5000:vmc_avg 0.396 / reason 0.407 / general 0.451 / doc 0.304 / ocr 0.40 / mmvp 0.51 / pope_acc 0.565 / pope_f1 0.694

| arm | vmc_avg | reason | general | mmvp | pope_acc | pope_f1 | 一句话 |
|---|---|---|---|---|---|---|---|
| **baseline** | 0.396 | 0.407 | 0.451 | 0.510 | 0.565 | 0.694 | 对照 |
| **sandwich**(自由 access,Q-img-Q) | ~0.396 | 0.447 | — | 0.55 | 0.617 | — | **幻觉+推理正向、无掉点(最佳 access)** |
| **windowearly**(强制早融合,img→q 层1-9) | 0.403 | 0.427 | 0.46 | 0.52 | 0.516 | — | 通用/推理小幅正,**但 POPE −4.9** |
| **auxexit**(sandwich+aux-exit k6) | 0.401 | — | — | ~base | ~base | — | 偏 vmc/doc(doc 0.328 最高) |
| **randpos**(随机位置) | 0.376 | — | — | — | 0.512 | — | 全程 ≤baseline,**有害** |
| **prefixlm**(全双向 prefix) | n/a | n/a | — | — | n/a | — | 推理 bug,数字不可用(causal 下 ckpt 健康) |
| **visualffn**(per-layer 视觉FFN expert,Mono-InternVL/BREEN式) | 0.403 | **0.433** | 0.434 | **0.537** | **0.598** | 0.708 | **全指标小幅正向,仅 general −0.017** |

visualffn 终点逐项 Δ vs baseline:vmc_avg +0.007 / reason **+0.027** / general −0.017 / doc +0.016 / ocr +0.010 / mmvp **+0.027** / pope_acc **+0.033** / pope_f1 +0.014。

## 两条核心结论

**① 架构层面:强制早融合(windowearly)未赢过自由 access(sandwich);visualffn 是唯一全指标无显著掉点的增强。**
- 让模型自由 access 并自行选择融合层(sandwich)优于强行压到早层(windowearly 用 POPE 换 general)。与 neo_report "中层融合是自发最优" 一致。
- visualffn(增强每层视觉处理容量)在终点带来**一致但小幅(+2~3 点)**的提升,最大涨点恰在最依赖视觉的任务(POPE 幻觉 +3.3、mmvp +2.7、reason +2.7),仅 general 掉 1.7。

**② 为什么提升都这么小 —— 基座视觉被 LLM 大幅忽略(根因诊断,2026-06-15)。**
视觉通路因果探针(`devtools/visual_pathway_probe.py`,visualffn ck5000,50 样本):
- 把图**涂黑** → 84% 答案不变;**换成别的图** → 88% 答案不变。
- 表征层 cos(真图,黑图)=0.51、跨图 0.37 → projection **确实编码了像素**(图物理上进了模型,排除深 bug)。
- 结论:**视觉通路是通的,但视觉特征质量太差、对齐太弱,LLM 学会了基本无视图、靠语言先验答题**(POPE 96.7% 恒答 yes / VMC 0.48 多靠文本先验)。训练 loss 在 step 500 即饱和,后 90% 步数几乎不再学。
- 这把"用这么多数据涨点这么小""加 visualffn 只 +0.7% vmc"统一解释了:**在优化一条被模型大幅忽略的通路,天花板被基座锁死。**

## 出路(按杠杆)
1. 大规模图文对齐预训练(对比/caption)再 SFT —— encoder-free 文献(EVE ~33M 对、Mono-InternVL ~十亿级视觉 token)都做、本实验跳过的一步。
2. 退一步上轻量视觉编码器(SigLIP/CLIP),直接拿高质量已对齐视觉特征。
3. 训练强制视觉依赖(需看图才能答的数据 / 削弱纯文本捷径)。

## 详情索引
实现/坑/读数:memory `early-fusion-access-arms-launch-2026-06`;诊断:memory `visual-pathway-diagnosis-2026-06`。
评测坑:watcher `submit_for` 硬编码 `--time=2h`,后期 vmcbench/POPE(慢节点+输出变长)系统性 TIMEOUT,需手动 3:30/4:00 重提。
