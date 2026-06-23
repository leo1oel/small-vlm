# VLM 融合分析方法手册:指标定义、合理性、测量代码与新模型接入

*本文档是可复现指南:每个指标给出 (a) 精确定义、(b) 为什么这样定义是合理的、
(c) 用哪个脚本怎么跑、(d) 解读纪律与已知坑。照此可在任何 decoder-only VLM 上
复现全部分析。英文正式表述见 `PAPER_ANALYSIS.md`,结论见
`FUSION_WINDOW_REANALYSIS.md`。*

---

## 0. 总览:三类正交的测量

| 轴 | 问题 | 指标 | 性质 |
|---|---|---|---|
| 计算分布 | 每层在加工哪一路 token? | u_S(ℓ) 逐流更新量 | 描述性 |
| 功能融合 | 答案在哪些层读取图像? | φ(d) 融合深度分布(注意力切断) | **因果** |
| 必要性 | 某一路的层内加工是否必需? | 流冻结保持率 | **因果** |
| 表征质量 | 图像特征何时达到 encoder 级? | 逐层 CKA 对冻结视觉 encoder | 描述性(带校准) |

核心方法论原则:**描述性指标(条带颜色、CKA)只能提示,结论必须由因果干预
(切断注意力 / 冻结残差流)背书**;且每个因果干预都配一个"换错图"对照,
把"模型坏了"与"图像没被用上"区分开。

---

## 1. 公共实验设置

- **数据**:VMCBench dev(1,000 道四选一 MCQ,20 类 × 50)。
  ⚠️ 该数据集**按类别排序**,任何子采样必须等距跨全集
  (`stride = n // n_causal`),否则前 N 条全是 MMMU/MathVista 难题,准确率会
  错得离谱(我们曾因此把 LLaVA 测成 0.36,真实值 0.52)。
- **打分**:单次前向,读答案位上 `A/B/C/D` 四个 token 的 logits,取 argmax
  (不做生成)。`letter_ids = {c: tok(c, add_special_tokens=False).input_ids[0]}`
  ——注意有的 tokenizer(如 SAIL 的 Mistral)首位是空格 piece,要取
  `input_ids[-1]`,务必打印检查。
- **换错图对照(donor swap)**:同一道题、把图换成第 `(i+37) mod n` 条的图。
  对所有干预都同时跑"真图"与"错图"两个版本。
- **深度归一化**:层号 ℓ 报告为 ℓ/N(N = decoder 层数),跨 28–48 层的模型可比。
- **token 分流**:`S_img` = `input_ids == image_token_id` 的位置
  (HF 模型从 `config.image_token_index` 或 `config.image_token_id` 读;
  自定义模型如 NEO/SAIL 从各自的 patch token id)。`S_txt` = 其余位置。
  ⚠️ 跑前必须 `assert vm.sum() > 0` 并打印 `n_vis`,id 错了所有干预会
  **静默变成空操作**,得到一条假的"无损"曲线。
- **注意力实现**:mask 干预的真正前提是 **hook 能拿到一个已物化的 4D mask**。
  HF 模型用 `attn_implementation="eager"` 保证这一点(SDPA 路径下 transformers
  可能判定 mask 平凡而传 `None`,hook 拿不到 mask 也不报错——干预**静默失效**)。
  SDPA 并非不可用:NEO(`patch_sdpa_attention()` 强制稠密 4D mask)和 SAIL
  (自带显式 4D prefix mask,模型代码不会跳过)就是 SDPA 下做手术的安全先例;
  但新模型默认走 eager 最稳。

---

## 2. 指标一:逐流计算量 u_S(ℓ)("分工条带")

### 定义
对层 ℓ、流 S ∈ {S_img, S_txt}:

    u_S(ℓ) = E_x [ mean_{i∈S} ‖h_ℓ(i) − h_{ℓ−1}(i)‖₂ / ‖h_{ℓ−1}(i)‖₂ ]

即该流 token 的残差向量在这一层被改写的平均相对幅度。画图时每条曲线除以
自身 L1..N 均值(看形状,不比绝对值)。

### 合理性
- 直接、模型无关、一次前向即可;`output_hidden_states=True` 不需要任何 hook。
- 相对幅度(除以入向量范数)消除了残差范数随深度增长的趋势项。
- **局限(必须写明)**:它测"笔画力度",不测"信息搬运"或"任务完成"。
  注意力把信息从 A 搬到 B 可以只给 B 加一个小范数分量;小更新也可能承重
  (NEO 冻结 L1-7 无碍、冻到 L11 即崩,而 L8-11 的 u_txt 仅 0.41≈均值)。
  所以条带**只用于描述分工形态,不用于下结论**——结论交给 §4 的冻结实验。
- 已知伪影:L0 有普适的"嵌入→残差"范数跃迁(可达 100–244×),最后 1–2 层
  有输出构建尖峰;归一化与解读都应排除 L0、不读末层。

### 代码
- HF 模型:`devtools/pathway_maturation.py <model_id> <kind> <out.jsonl> [n]`
  (kind ∈ {llava, qwen, gemma, onevision, llavanext},新增模型加一个分支即可)
- NEO:`neo_analysis/neo_pathway_maturation.py <out.jsonl> [n] [stage]`
- SAIL:`sail_analysis/sail_pathway_mat.py <out.jsonl> [n]`
- 输出 jsonl 字段:`u_img/u_txt/u_last/norm_img/norm_txt`(每层一个值)。

---

## 3. 指标二:融合深度分布 φ(d)(主指标,因果)

### 定义
1. **可用图像信号** R₀ = A(0) − Ã(0),其中 A 为字母准确率,Ã 为换错图版本,
   d=0 表示不干预。R₀ 是"正确的图比错误的图多带来的准确率",天然扣除了
   语言先验与格式效应。
2. **prefix 切断**:在层 [0, d) 内屏蔽所有 *text-query → image-key* 注意力
   (4D mask 上把 [text 行 × image 列] 置 −∞;image↔image、text↔text 不动),
   得 A(d)、Ã(d);**保留信号** r(d) = [A(d) − Ã(d)] / R₀。
3. **融合深度分布**

       φ(d) ∝ max(0, r(d−1) − r(d)),   Σ_d φ(d) = 1

   φ(d) = 可用图像信号中"在第 d 层变得不可恢复"的份额,即承重的跨模态
   读取发生在哪一层的分布。报告其分位数:**中位 q50、箱 [q25,q75]、
   须 [q10,q90]**(在 φ 的单调 CDF 上取,无任意阈值、抗噪)。

### 合理性
- **因果而非相关**:直接禁用跨模态信息通道,看答案损失多少——这是"融合"
  的操作化定义本身,不是代理量。
- **R₀ 归一化**使强弱模型可比:测的是"图像信号的使用位置",与模型总准确率脱钩。
- **错图对照在每个 d 上都跑**:r(d) 的分母分子同受干预,屏蔽掉"切注意力
  本身造成的非特异损伤"。
- **分位数优于阈值**:曲线噪声 ±0.05–0.1,"跌破 90%"这类阈值会被平台期
  噪声击穿;CDF 分位数稳定(q50 的 bootstrap 95% CI 在 nc≥166 时 ±0.03–0.08)。
- **必须知道它不测什么**:prefix 切断测"最晚必须在哪读"(电路依赖位置)。
  若担心"平时早读、被切后由后层补救"的冗余假象,用 §3.1 suffix 检验。
- 稳健性清单(我们全部做过、应随结论报告):NLL 版 r(d)(消字母 argmax 偏置)、
  按题类分桶(检验是否模型属性)、bootstrap CI、与注意力分配深度质心交叉验证。

### 代码
- 万能探针:`devtools/fusion_window.py <model_id> <kind> <out.jsonl> [n] [n_causal] [sweeps] [rowscope]`
  - `sweeps`: `prefix` | `suffix` | `both`;`rowscope`: `alltext`(默认)| `lastrow`
  - 输出:`intact/swap` 全样本;causal 子集多 `cost[k]`(=切 [0..k+1))、
    `suf[k]`(=切 [k..N))
- mask 手术核心(简化示意;实际实现见 `fusion_window.py` 的 `_pre`+`_isolated`,
  另有两点工程细节:① 改写后的 mask 按 `id(mask)` 缓存、换图/换深度时
  `_mc.clear()`,避免每层重算;② `rowscope=="lastrow"` 时 `tp` 只取最后一行,
  即 §3.2 的通路分解变体):

```python
def _pre(self, i):                       # forward_pre_hook(with_kwargs=True)
    def f(_m, args, kwargs):
        active = ((self.mode == "prefix" and i < self.depth) or
                  (self.mode == "suffix" and i >= self.depth))
        if active and self.vm is not None:
            am = kwargs.get("attention_mask")
            if am is not None and am.dim() == 4:   # eager 下恒为 4D
                m = am.clone()
                if self.rowscope == "lastrow":     # 只切答案位那一行(§3.2)
                    tp = torch.tensor([m.shape[-2] - 1], device=m.device)
                else:                              # 切全部 text 行(默认)
                    tp = (~self.vm).nonzero().squeeze(-1)
                vp = self.vm.nonzero().squeeze(-1)      # image 列
                neg = False if m.dtype == torch.bool else torch.finfo(m.dtype).min
                m[0, :, tp[:, None], vp[None, :]] = neg
                kwargs = dict(kwargs); kwargs["attention_mask"] = m
                return args, kwargs
        return None
    return f
```

- 分析:`devtools/window_analysis.py`(窗口表+图)、分位数/箱线计算见
  `devtools/fig_two_conclusions.py::fusion_quants`。
- **正确性不变量(每个新模型必须验证)**:
  `suf[0] ≡ cost[N-1]`(都是全切断,确定性前向下应逐位相同);
  `suf[N-1] ≈ intact`;`r(d)` 起点 ≈1、`retained_suf(0)` ≈0。

### 3.1 suffix 切断(融合起点)
切 [d, N):`retained_suf(d)/R₀` = **前 d 层已经搬进 text 流的可用信号量**
(搬进去的信号此后可经 text→text 传播)。上升点 = 融合实际开始的下界紧测度,
专治"质心/阈值掩盖早融合"与"冗余补救"两类质疑。同一脚本 `sweeps=suffix`。

### 3.2 lastrow 变体(通路分解)
`rowscope=lastrow` 只切**答案位那一行**的图像读取,保留 question token 读图。
对照全行版本可判定信号走 image→question→answer 还是 image→answer 直读
(我们测得所有模型直读通路承重 ≤9%)。

---

## 4. 指标三:流冻结(必要性检验,因果)

### 定义
freeze_S@k:对层 **[1, k)**(⚠️ 必须保留第 0 层,见坑 §6.3)的每一层,把
S 流位置的层输出改写回该层输入——S 的残差流被钉在 L0 值直到第 k 层,
其它一切(另一流的计算、对被冻值的注意力读取、sink 形成)照常。报告
**保持率 = R₀(frozen)/R₀(baseline)** 随 k 的曲线,或 k≈0.2N 处的单值。

### 合理性
- 与 §3 互补:§3 禁"读",本节禁"写"。两者组合才能把
  "图像在哪被读"与"图像是否需要先被加工"分开。
- **解读纪律(对抗审查结论)**:只有"冻到 k 仍无损"是干净证据
  (该加工不必要);深 k 处的损坏是分布外冲击的上界,不能反推"加工必要的
  具体层位"。我们的结论只用无损方向 + 同深度跨模型对比(双重分离)。
- text 冻结会连带冻结答案位(其计算深度被压缩),跨模型比较时注意;
  对"NEO pre-Buffer 文本直通"这类断言无碍(答案位也是 text、同样该直通)。

### 代码
- `devtools/freeze_probe.py <model_id> <kind> <out.jsonl> [n] [n_causal]`
  (HF);`neo_analysis/neo_freeze_probe.py`;`sail_analysis/sail_freeze_probe.py`
- 实现:forward_pre_hook 存输入,forward_hook 上 `h[0, pos] = stored[0, pos]`;
  `pos` = S 的位置(text 冻结剔除位置 0,保护 sink)。
- 分析:`devtools/freeze_analysis.py`(曲线)、`devtools/fig_freeze_bars.py`(摘要)。

---

## 5. 指标四:逐层 CKA 对冻结视觉 encoder(表征质量)

### 定义
对每层 ℓ,取图像 token 表征的 mean-pool,得 X_ℓ ∈ R^{n×d}(n 张固定图);
对参照 encoder(DINOv2,**所有被测模型都没用过它**,故中立)同图取
patch-token mean-pool 得 Y。报告 linear CKA(X_ℓ, Y) 随 ℓ/N 的曲线
(Gram 形式 O(n²d),n=100 时秒级)。

### 合理性
- 回答"图像特征何时达到 encoder 级":native 模型曲线从 raw-patch 水平爬升、
  在其预编码结构边界处触顶,且顶点高度 ≈ 真 encoder 输出进入 LLM 的水平
  (Qwen ViT 入口 0.69 vs NEO 峰值 0.68/0.73)——这是"pre-Buffer ≈ encoder"
  的表征证据(`fig_prebuffer_cka`)。
- **内置校准**:LLaVA 图像 token 对它自己的 CLIP encoder 入口 CKA=0.89,
  证明该度量确实能识别"就是这个 encoder 的特征"。
- **局限**:对单一参照的 CKA 是代理量,读轨迹形状不读绝对值
  (裸像素对 DINO 也有 ~0.6 的天然相关,见 SAIL 入口);跨参照
  (DINOv2/SigLIP/CLIP)检查结论是否依赖参照选择。

### 代码
- 抽取:`devtools/cka_extract.py <kind> <model_id> <tag> [N]`(VLM 与参照
  encoder 同一脚本;NEO/SAIL 各有专用版),存 `cka_<tag>.npz` (L+1, n, d)。
- 计算+图:`devtools/cka_numbers.py`(数字)、`devtools/cka_compute.py`、
  `devtools/fig_prebuffer_cka.py`。⚠️ 必须用 Gram 形式
  `CKA = ⟨K̃,L̃⟩/(‖K̃‖‖L̃‖)`,特征空间形式在 d=4096 时慢 3 个数量级。

---

## 6. 新模型接入清单(按顺序做)

1. **接入分支**:在 `fusion_window.py` / `pathway_maturation.py` /
   `freeze_probe.py` 加 kind 分支(基本只需 `from transformers import XXX as M`
   + chat template 的构造方式)。`find_layers()` 已覆盖四种常见 decoder 路径。
   ⚠️ 当前 kind 覆盖不对称:`fusion_window`/`pathway_maturation` 支持
   {llava, qwen, gemma, onevision, llavanext},`freeze_probe` 只有
   {llava, qwen, gemma}——给新模型跑冻结实验要自己补分支。
2. **冒烟(n=8)**:打印 `N_layers / img_id / n_vis / letter_ids`;肉眼核对
   5 条样本的 pred vs gt;intact 准确率对照该模型的公开 benchmark 数
   (LLaVA-1.5 ≈ 0.52 这一步抓出过子采样 bug)。
3. **不变量**:`suf[0] ≡ cost[N-1]`(逐位)、`retained_suf(0)≈0`、
   `r(小 d)≈1`、`R₀ > 0.05`(R₀ 太小一切归一化曲线都是噪声,先查 prompt/打分)。
4. **全量**:n=1000、n_causal=200–334(等距);u_S 用全 1000。
   单卡 L40S:7B 模型 prefix+suffix 全套约 2–4 h;>10B 或 >4k image token
   用 A100/H200 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
5. **稳健性**:bootstrap CI(≥200 重采样)、NLL 版、按类分桶。
6. **报告**:q10/q25/q50/q75/q90 + R₀ + nc;分工条带;(可选)冻结保持率。

## 6.3 已知坑(每个都付过学费)

| 坑 | 症状 | 解法 |
|---|---|---|
| SDPA/FA2 下 mask 为 None | 干预静默失效,曲线平坦"无损" | 强制 eager;hook 里对"本应命中却没命中"计数报警 |
| 冻结含第 0 层 | freeze_txt 对所有模型灾难性(平凡伪影) | 冻结窗口 [1,k),保留 L0 范数跃迁 |
| 数据集类别排序 | 子集准确率严重偏低/偏置 | 等距 stride 采样;对比公开数 sanity check |
| image token id 取错 | vm 全 False,干预空操作 | assert n_vis>0 + 启动时打印 |
| letter id 取错(空格 piece) | 打分全错 | 打印 letter_ids,必要时取 input_ids[-1] |
| OOM 跳样本有偏 | causal 子集类别构成漂移(Qwen 的 DocVQA 全跳) | 记录 skip 原因,报告时核对类别分布 |
| R₀ 太小 | 归一化曲线全是噪声 | 先修 prompt/打分,R₀<0.05 不报曲线 |
| resume 改 n/n_causal | donor 配对与分层错乱 | 续跑参数必须与首跑完全一致 |

---

## 7. 文件索引(一行命令模板)

```bash
# 分工条带(全 1000)
python devtools/pathway_maturation.py <model_id> <kind> out_mat.jsonl 1000
# 融合窗口(prefix+suffix,分层 causal)
python devtools/fusion_window.py <model_id> <kind> out_win.jsonl 1000 200 both
# 通路分解(只切答案位)
python devtools/fusion_window.py <model_id> <kind> out_lr.jsonl 1000 200 prefix lastrow
# 流冻结
python devtools/freeze_probe.py <model_id> <kind> out_frz.jsonl 1000 150
# CKA(VLM 与参照各跑一次)
python devtools/cka_extract.py <kind> <model_id> <tag> 100
python devtools/cka_extract.py dino facebook/dinov2-base dino 100
# 汇总与图
python devtools/window_analysis.py        # 窗口表 + fig_window/fig_lastrow
python devtools/freeze_analysis.py        # 冻结曲线 + fig_freeze
python devtools/fig_two_conclusions.py    # 两张单结论主图
python devtools/fig_dol_pretty.py         # 总分工地图
python devtools/fig_prebuffer_cka.py      # pre-Buffer ≈ encoder
```

NEO/SAIL 在 neo venv(transformers 4.57 + pyarrow 读数据集),其余在主
.venv(transformers 5.10);SLURM 模板见 `devtools/*.slurm`(ckpt-all/cse-ckpt,
L40S 默认,大模型 a100/h200)。
