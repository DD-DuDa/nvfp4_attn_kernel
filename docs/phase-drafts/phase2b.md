# Phase 2b Draft — warp 布局重划分

## 证据与约束

- 高 batch IKET: softmax0 launch-tail; correction 89.8% wait softmax; MMA 79.4% wait P; epilogue 99.6% wait correction。前置条件成立。
- 先只回收 warp15；再决定 softmax stage。任何跨 warp 行拆分必须同轮有协作归约和数值 gate。
- MHA Phase2回归 0.911x geomean / 0.692x worst 必须携带，不得隐藏。
- 纯 FP4 page-aligned 主线；BF16-Q residual regressions保持绿。
- 每轮 numerical + clean performance + IKET；≤10轮或2小时。
- 内部 Terra/high 主审 + 另一模型交叉审。

### Phase 2b — warp 布局重划分

对应 §5.5。**前置条件：Phase 0b 的 IKET trace 显示 softmax 或 correction 角色确实
占据关键路径。** 如果关键路径在 load 或 MMA 上，本 Phase 直接跳过——不要因为论文
说得有道理就动手，那正是 §6.3 要杜绝的无证据轮次。

本 Phase 受 BitDecoding 启发最深，因此重申 §6.2 的定性：**那是知识库不是规范。**
`W_m=1 / W_n=4` 是 SM80/SM90 上 `mma.m16n8k16` 的结论，SM100 的 `tcgen05` 由单
warp 发射 MMA，不存在同构的搬运。可借用的是「decode 场景下 M 轴不值得分 warp」
这个判断和 6.1x 的量级预期，具体改法必须自己按 IKET 证据推。

排在 Phase 2 之后有个实际理由：Phase 2 去掉 Q padding 之后，M 轴才真的只剩
`g_q` 行，此时「8 个 warp 分在 M 上」的浪费才完全暴露，重划分的收益也才好度量。
顺序反过来会把两件事的效果搅在一起。

分两步，中间可验证：

1. **先只回收空转的 warp 15**，这是最小改动、无正确性风险的一步，用来校准「多一个
   warp 值多少」。
2. **再动 softmax 的 stage 划分**。一旦 softmax 的行被拆到多个 warp 上，就必须同时
   引入跨 warp 归约（BitDecoding 的 `sTMP`）和 P 的 smem 中转（`sAcc`），否则
   rowmax 不完整、P 的分布也对不上 PV MMA 的 operand 布局。

**验收**：
- IKET 上关键路径角色的等待占比下降，且 16 个 warp 的 lifetime 离散度收窄
- **数值 gate 必须与性能 gate 同轮跑**——这是本 Phase 的硬性要求，理由见下
- 全表不退化，包括低 batch

> **本 Phase 最大的风险是「快而错」。** BitDecoding Table III 的中间行记录了这个
> 陷阱：拓宽 N 但不加协作 softmax，延迟从 3.746 ms 降到 0.610 ms（6.1x），TC 利用率
> 从 10.91% 升到 19.71%，**结果是错的**。任何只看性能的门槛都会把它判为大胜并合入。
> 所以本 Phase 不接受「先记下来后面修数值」的轮次。
