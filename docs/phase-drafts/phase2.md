# Phase 2 Draft — 高 batch 去除 Q padding / 冗余 CTA

## 不可变约束

- 先纯 FP4 Q/K/V、无 residual、`len % 128 == 0`；BF16-Q residual 回归仍必须绿。
- 目标区间 batch >=16；D0/D3/D4/D7 不变。
- Phase 0 trace 的 baseline case 显示 softmax0 launch-tail，MMA 约 68% 等 P-ready；baseline grid 是 16 CTA/16 SM。Phase 2 高-batch配置需重新取证，不能把 baseline CTA 比例外推。
- 着手点是去掉 `_kernel.py` 的 128 行 query padding，不是启用 pack_gqa。
- 每轮 clean performance + numerical gate + IKET；流量定量才用 ncu。
- 预期：GQA-4/8 下 padding 导致多余 M tile/CTA；去掉后高 batch clean latency预期 1.5–4x，实际远低需原因分析。
- ≤10 轮或 2 小时；失败记录后继续 Phase 2b/3；主审 Terra/high + 另一内部模型。

## 原始 Phase 内容

### Phase 2 — 高 batch：消除冗余 CTA 与冗余 KV 流量

对应 §5.1。目标区间 `batch ≥ 16`，那里瓶颈是每 CTA 吞吐。

取证分两层，**先 IKET 后 ncu**：

1. **IKET 先答结构问题**，且不需要计数器。trace 的 launch 记录直接给出 `gridDim`，
   `locationTable` 给出实际跑起来的 CTA 与 SM 分布。「每个 KV 头起 4 个 CTA」这个
   断言看 grid 就能证实或证伪。同时看 load warp（14）是不是关键路径、它的等待占比
   多少——如果 load 角色饱和而 MMA 角色大量空等，冗余流量假设就成立。
2. **ncu 只用来定量**：把冗余量化成字节。预期表现为 L2/DRAM 流量约为理论 KV 字节数
   的 4x。这一步 IKET 答不了，因为它没有任何流量计数器。

假设不成立就回到 IKET 的角色表重新定位，而不是硬推原方案。

按 §5.1 的更正，本 Phase 的着手点是**去掉 Q 的 128 行 padding**，不是启用
`pack_gqa`（它本来就开着）。改动会波及 `_kernel.py` 的缓冲分配、`_decode.py` 的
布局适配、以及 kernel 内 `seqlen_q_static` 的取法
（`fp4_decode_kernel.py:1906`）——比「翻一个开关」大得多，排期要按这个体量估。

**验收**：`batch ≥ 16` 区间几何平均相对 Phase 1 提升；IKET 上关键路径角色的等待
占比下降；ncu 实测 DRAM 流量接近理论 KV 字节数。数值门槛不变。
