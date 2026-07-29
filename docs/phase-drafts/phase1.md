# Phase 1 Draft — 新输入合同（q_fp4 直通）

## 不可变约束

- D2：保持单入口 `fp4_decode`，新增可选 `query_fp4 + query_scales`；BF16 与 FP4-Q 双路径共用 core。
- Phase 1 必须新增 FP4-Q 数值测试；不得改数值门槛。
- Phase 1 失败必须整体停下等用户。
- 所有 shell 显式设置 §6.1 PATH 和 `CUDA_VISIBLE_DEVICES=1`。
- 禁止外部 `codex exec`/`codex review`；主审 Terra/high，另一内部模型交叉 review。
- Phase 预算 ≤10 轮或 2 小时；以 `tests/kernel` 全绿 commit 收尾。
- 开工预期：预量化 Q 消除每调用 Q quantize/scratch 开销。Phase 0 breakdown 中 Q quantize 为 0.00611 ms、约 11.15% 的 profiler GPU composition；预期短 case clean GPU latency改善约 5–15%，核心 decode 输出与 BF16-Q 路径逐字节一致。

## 原始 Phase 内容

### Phase 1 — 新输入合同（q_fp4 直通）

按 D2 实现单入口双路径。这是目标的前提，也是最便宜的一笔收益。

**本 Phase 是唯一「失败即整体停下」的 Phase**（§6.6 第 4 条）：后面所有测量都在
新合同上做，它不成立则后续数字没有意义。

**验收**：
- 与现有 BF16-Q 路径在同一份输入上数值一致（先用现有 quantizer 产生 `q_fp4`，
  再喂新入口，结果应与旧路径逐字节相等）
- **`tests/kernel/` 新增 FP4-Q 入口的数值测试**（D2 的绑定要求）。不加的话新入口
  没有任何 gate，而它正是性能目标所在的路径
- `tests/kernel/` 全绿（含新增用例）
- 记录加速幅度，FP4-Q 与 BF16-Q 两条路径分别记——这是后续所有 Phase 的新起点
