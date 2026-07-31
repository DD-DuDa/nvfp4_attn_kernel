# Phase 11 — 混合路径重建、TMEM 解耦、与 trtllm-gen 的外部对照

Phase 10 让纯 FP4 路径达标（D3，s16384 graph 几何平均 0.479），但门槛之外留了两个
缺口：混合路径 3.524，MQA 0.858。本阶段做完三件事——把混合路径拉回达标线内、把 P/V
的 scale factor 从 S 的 tensor memory 里挪出去、以及第一次拿外部实现做对照。

所有性能数字都来自 CUDA graph replay。图捕获会把 host dispatch 从两边一起抹掉，
eager 计时会按各自走的内部路径分别收费，不是门槛想度量的东西。

---

## 1. 混合路径：3.524 → 0.632

四个提交，每个自带绿色的 `tests/kernel`。

| 提交 | 改动 | s16384 hybrid 几何平均 | b1 |
|:--|:--|---:|---:|
| — | Phase 10 结束态 | 3.524 | — |
| `0aa7c43` | 转置 softmax 覆盖 residual，靠 SMEM 做种子桥接 | 1.869 | 121 us |
| `fcdea9e` | residual 不再禁用 split-K | 1.435 | 55.4 us |
| `58ce666` | residual 的 QK 与 P 一并转置 | 0.715 | 22.7 us |
| `f57fd61` | SFP/SFV 迁出 S 的 TMEM 列 | 0.618 | — |
| — | 今日复测（`graph-gate-hybrid.json`） | 0.632 | 20.9 us |

s65536 同步测到 0.476。两档都在 0.5 门槛内，而 Phase 10 时是 3.524 / 3.447。

### 1.1 `0aa7c43` — 种子桥接

residual 是一块不转置的 BF16 tile。当时选择不动它，只把它的 softmax 种子换向，
每个 tile 经 SMEM 一次。最大值可以在线程间复制，和不能：转置之后一个线程持有的是
kv 位置，tile 结束时和是一个跨线程的求和。

顺带暴露两个潜伏缺陷。BF16 的 PV MMA 一直在继承 FP4 P 的 operand source，而转置已经
把 P 挪到了 SMEM；转置版的 seqlen mask 少了把 block −1 折到 block 0 的 clamp，只有
「完全没有 fp4 页」的行才能走到，也只有 residual 能造出这种行。

### 1.2 `fcdea9e` — 慢的是并行度，不是 softmax

带 residual 就整体禁用 split-K，于是单行解码 128k 上下文只跑 8 个 CTA（每 kv head
一个），每个走完整条序列。

修法不是按 split 索引给流水线阶段开分支，而是**每个 split 都跑 residual 块，非归属的
split 拿到长度 0**。零长 residual 行本来就是合同的一部分，所以这个 no-op 只多一次
BF16 页加载，其余一切都不会走岔。

为此挪了两处。一是「完全没有 fp4 页」的行——只有 residual 造得出来——仍必须运行拥有
residual 的那个 split，而六个 warp 各自独立判断 tile 有没有活干；现在统一问
`tile_has_work` 一处。二是 SMEM 预算：epilogue ring 的档数超过了索引它的 q tile 数，
而 `sQ` 被加宽去覆盖 `sO` 时又漏算，两头夹着放不下 BF16 缓冲。

### 1.3 `58ce666` — 寄存器溢出才是主项

residual 是旧朝向的最后一个消费者。它的 softmax step 在转置主循环的状态还活着的时候，
把整块 128×128 的分数 tile 摊成每线程 128 个 fp32。128 寄存器/线程放不下：ncu 量到
**18.5 MB 的 local memory 溢出流量**，纯 FP4 路径是零；iket 把 88 us 里的 70 us 记在
这一个 step 上。

把 BF16 QK 的两个 operand 对调，residual 的分数就和 fp4 块同向落下，一套 softmax 读
两者，每线程的 fragment 收缩到活跃查询行数。P 随后和 fp4 侧一样从 SMEM 进 tensor
core，复用 residual Q tile 的字节（QK GEMM 在写 P 之前就释放了它）。种子桥接要跨的
那个朝向差随之消失，函数一并删掉。

### 1.4 `f57fd61` — QK 被 PV 串在后面

P 和 V 的 scale factor 放在 S 刚被读走的那些列上，于是下一块的 QK 必须等这一块的 PV
消费完，而 PV 等 softmax。等于把 QK 的延迟塞进了两个 softmax step 之间。

转置之后 P 本身住在 SMEM，只剩 scale factor 还在混用，而 decode 在 O 之上留了 128 列
tensor memory 没人用。挪过去以后 S 成了下一个 QK 唯一会覆盖的东西，softmax 在它的
load 退休时就能释放 S，而不是等整个 step 结束。QK 就能在等 P 之前发出，跑在 softmax
底下而不是它后面。

这一刀对**纯 FP4** 同样有效——见 §2。

> 记录里有一处未对齐：`fcdea9e` 记「纯 FP4 保持 0.478」，`f57fd61` 记纯 FP4 从
> 0.606 改善到 0.380。0.478 与 0.606 之间的落差没有复现过，未查。今日复测的 0.381
> 与 `f57fd61` 记的 0.380 一致。

---

## 2. 纯 FP4 现状

`docs/perf/phase11/graph-gate-long.json`，对 FA4 的 graph 几何平均：

| seqlen | Phase 10 | 今日 | 相对 FA4 |
|---:|---:|---:|---:|
| 16384 | 0.479 | **0.381** | 2.62x |
| 32768 | 0.475 | **0.352** | 2.84x |
| 65536 | 0.464 | **0.363** | 2.75x |
| 131072 | 0.470 | **0.342** | 2.92x |

D3 门槛是 0.5，现在四档都在 0.35 附近。

---

## 3. Prologue 与首块 KV 的头部代价

`docs/perf/phase11/iket-prologue-b1-s16384.{txt,json}`，b1/s16384 纯 FP4，load warp
（插桩状态下 kernel wall 13.44 us；同一形状的干净 graph 时间是 10.5 us）：

| 段 | 每 CTA 均值 | 内容 |
|:--|---:|:--|
| `pro_init` | 183 ns | TMA 描述符预取、mbarrier 初始化 |
| `pro_sync` | 126 ns | KV 流水线构造里的 fence + CTA 同步 |
| `pro_setup` | 26 ns | 类型与 tile scheduler 构造 |
| `load_seqinfo` | 248 ns | 读 `seqused_k` |
| `load_pageidx` | 241 ns | 页表查找 |
| `load_issue_k0` | 128 ns | 发出第一个 K 的 TMA |
| **合计** | **952 ns** | 第一个 KV 字节被请求之前 |

结论是：**setup 本身只占 335 ns，剩下的是一条指针追逐**——`seqused_k` 要先回来才知道
读页表的哪一行，页表要先回来才知道 TMA 的地址。三次串行的 DRAM 往返在 b1/s16384 的
10.5 us 上占 9%，seqlen 越短占比越大。

可动的是前两跳：`seqused_k` 能提到 kernel 最开头、与描述符预取重叠，页表行可以在同一
次访问里预取。第三跳是冷 DRAM 读，不可约。

---

## 4. 本轮 IKET 暴露的新瓶颈：行最大值归约

同一份 trace，softmax warp 的分项（每 KV 块均值）：

| 分项 | 均值 | 占 softmax 生命 |
|:--|---:|---:|
| `sm_rowmax` | 806 ns | 27.8% |
| `sm_pquant` | 705 ns | 24.3% |
| `sm_wait_s` | 284 ns | 9.8% |
| `sm_exp` | 177 ns | 6.1% |

`exp2` 本身已经是最小的一项。**跨线程的行最大值蝶形现在是 softmax 里最贵的，是 `exp2`
的 4.5 倍**——这是转置的直接代价：行最大值从线程本地变成了跨线程归约。

消费侧一致：`mma_wait_p` 占 mma 生命的 50.9%（均值 1.56 us），`corr_wait_sm` 占
correction 生命的 76.6%。整条链仍然被 softmax 拖着，但拖住它的已经不是指数运算。

---

## 5. 外部对照：vLLM / FlashInfer 的 trtllm-gen NVFP4 decode

### 5.1 方法

`tests/kernel_profile/bench_vs_trtllm.py`。两边都在 CUDA graph replay 下计时，计时区间
内不含任何量化：K/V 事先量化成各自 kernel 要求的 NVFP4 分页格式，Q 也事先量化。页大小
两边都是 128。

选中的 cubin 是 `fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvDense...`，确认走的是
NVFP4 那条。不对称的地方有两处，都属于两个 kernel 的固有差别：trtllm-gen 的 Q 是 FP8
E4M3、输出只能是 FP8（要 BF16 输出直接报 `Missing TRTLLM-GEN kernel`），我们的 Q 是
FP4 E2M1、输出 BF16。Q 和 O 的字节量在长序列下可忽略，KV 与 scale 的流量两边完全相同
（每 16 个元素一个 E4M3 scale）。

页大小扫过 16/32/64/128 确认没有低估对方，128 是它最快的一档，四档差异在 1% 以内。

### 5.2 长文本，GQA 32:8（`docs/perf/trtllm/longctx-gqa4.json`）

比值 = 我们 / trtllm，越低越好。

| seqlen | b1 | b4 | b16 | b64 | 几何平均 |
|---:|---:|---:|---:|---:|---:|
| 16384 | 0.63 | 0.84 | 0.87 | 0.92 | **0.806** |
| 32768 | 0.70 | 0.92 | 0.89 | 0.88 | **0.844** |
| 65536 | 0.85 | 0.97 | 0.89 | 0.96 | **0.915** |
| 131072 | 0.91 | 0.98 | 0.87 | 0.98 | **0.933** |
| 262144 | 1.00 | 0.96 | 0.95 | — | **0.969** |

序列越长优势越薄。原因直白：两边都撞在 DRAM 上。平台带宽我们 5.3–5.6 TB/s，对方
4.9–5.2 TB/s，差的就是这 8%。b1/s262144 那格只有 4.47 TB/s，是单行长序列并行度没铺满。

### 5.3 换 head 配置（`docs/perf/trtllm/longctx-heads.json`）

这里翻车。

| 配置 | GQA group | 我们平台带宽 | trtllm | 比值范围 |
|:--|---:|---:|---:|:--|
| 32:8 | 4 | 5.4 TB/s | 5.1 | 0.81–0.97 |
| 32:4 | 8 | 4.0 TB/s | 4.9 | 1.06–1.35 |
| 64:8 | 8 | 4.1 TB/s | 5.1 | 0.89–1.27 |
| 32:1 (MQA) | 32 | 1.7 TB/s | 2.3 | 1.34–3.03 |
| 32:32 (MHA) | 1 | — | 无 kernel | trtllm 跑不了 |

解释变量是 **GQA group size（`qhead_per_kvhead`），不是 `heads_kv`**：32:4 与 64:8 的
group 都是 8，带宽都落在 4.0–4.1，而 group 4 是 5.4。我们的平台带宽随 group 增大明显
下滑，对方守住 5 TB/s 直到 MQA 才崩。这套 kernel 是围着 group 4 调出来的。

MHA 那档 trtllm 报 `Missing TRTLLM-GEN kernel`，它的 NVFP4 decode 没覆盖 `heads_kv=32`。

### 5.4 尚未查清

`transpose_s` 的启用条件只要求 group 是 2 的幂且 ≤128，group 8/32 都走了转置路径，
所以不是快路径没生效。两个待验证的怀疑，本阶段未做：

- MQA 的 b1 只有 0.36 TB/s，那是并行度——`heads_kv=1` 时未拆分只有 1 个 CTA，split-K
  上限 32。
- 64:8 的 b64 有 512 个 CTA 仍只有 4.1 TB/s，那就不是并行度。更像是每 CTA 内随 group
  线性增长的 softmax 成本没被带宽掩盖住，与 §4 的 `sm_rowmax` 指向同一处。

---

## 6. 验收

`PYTHONPATH=src python -m pytest -q tests/kernel` → **58 passed**，且是在 IKET 埋点
在树内的状态下跑的（D1 要求「插桩后仍过 `tests/kernel`」）。

## 7. 产物

| 文件 | 内容 |
|:--|:--|
| `docs/perf/phase11/graph-gate-long.json` | 纯 FP4，四个长 seqlen × 四个 batch |
| `docs/perf/phase11/graph-gate-hybrid.json` | 混合路径，s16384 / s65536 |
| `docs/perf/phase11/iket-prologue-b1-s16384.{txt,json}` | 头部代价与 softmax 分项 |
| `docs/perf/trtllm/longctx-gqa4.json` | 对 trtllm-gen，32:8，到 256k |
| `docs/perf/trtllm/longctx-heads.json` | 对 trtllm-gen，四种 head 配置 |
| `docs/perf/trtllm/nvfp4-vs-trtllm.json` | 首轮对照，含 s1024 / s4096 |
| `tests/kernel_profile/bench_vs_trtllm.py` | 对照脚本 |
