# Phase 0 Draft — 度量基建（先 0b 后 0a）

## 不可变约束

- 本 draft 派生自 `docs/tasks/2.fp4_decode_speedup.md`；§12 的 D0–D8 均不可修改。
- 执行顺序必须是 **Phase 0b 先于 Phase 0a**。
- 所有 shell 先设置 vllm-nvfp4 PATH 和 `CUDA_VISIBLE_DEVICES=1`。
- 性能数字只来自干净 run；IKET 插桩 run 只用于核内证据。
- 0b go/no-go 最多 2 轮；整个 Phase 最多 10 轮或 2 小时。
- Phase 收尾必须 `tests/kernel/` 全绿并提交。
- 预期：Phase 0 不改 kernel 箭法，性能应不退化；同配置三次几何平均波动 <3%。IKET 应识别关键路径角色、PV 前序列化与 CTA/SM 铺开。

## 原始 Phase 内容

### Phase 0 — 度量基建（不改 kernel 算法）

没有可信的度量，后面所有轮次都是盲跑。本 Phase 分两半：benchmark 侧和 IKET 侧。

#### 0a. Benchmark 侧

- **补齐 `batch=64/128 × seqlen=65536` 两格**：把 `--max-kv-tokens` 从默认
  2,200,000 提到 ≥8,400,000。这两格不是跑不下（BF16 KV 32 GiB，卡上 191.5 GB），
  是被这个旋钮挡掉的。前提要实测确认：若 bench 同时给四个 variant 分配输入，峰值
  可能到 60–70 GiB，仍在容量内但不要假定。**这一步必须先于任何门槛判定**，它把
  `seqlen=65536` 的 gate 从 15.7x 降到约 10.5x
- 修好 per-kernel 拆分：`results.json` 的 `kernels` 字段目前全空，
  拿不到 Q 量化 / attention / epilogue 的分项耗时
- 把 §5.4 的 Q 量化占比测出来
- **同时记录 FP4-Q 路径与 BF16-Q 路径两个数字**（D4）。gate 认前者，后者用来随时
  拆开「接口变更白送的」与「kernel 真改出来的」两笔账
- 扩展 grid：至少加 `batch ∈ {2, 8, 32}`，以及 GQA 之外的 MHA/MQA 各一组
  （目标说的是「任意 batch, seq_len」，只测 GQA-8 覆盖不了）
- 输出：几何平均 + 最差单点，两个数一起报
- 固化基线定义（D0），把 FA4 的 `num_splits` 记进结果
- **给结果加 provenance**：env 路径、python / cutlass / flash-attn 版本、GPU 型号、
  commit。现在 `results.json` 是个裸 list，什么都没记；这次靠「torch-base 没装
  flash-attn」才反推出它是哪个环境跑的，纯属侥幸。RLCR 会产出几十轮数据，没有
  provenance 就无法回溯。

#### 0b. IKET 侧

> **先做 go/no-go，预算 2 轮。** IKET 目前只在一个 64 线程、2 block 的玩具 kernel
> 上验证过（`/tmp/iket_smoke/smoke.py`），**从未跑过 `fp4_decode_kernel.py`**——
> 一个 4700 行、16 warp 特化、带 TMA/UMMA/TMEM 三级流水的持久化 kernel。两者之间
> 隔着几类已知会出问题的东西：16 warp × 长序列的事件量可能撑爆 buffer、marker 必须
> 落在 warp-uniform 位置（本 kernel 满屏 `if warp_idx == ...` 分支）、以及与 mbarrier
> 流水同步的相互干扰。
>
> **过了**，按下文 IKET 主、ncu 备执行。**没过**，主备顺序对调：ncu 转正，§8.3
> 的证据要求改成「IKET 或 ncu」。
>
> 这一步存在的理由是 §8.3 写了「无核内证据的轮次视为无效」。若 IKET 在真实 kernel
> 上跑不通而不设退路，**这条规则会把整个循环锁死**——每轮都无效，agent 空转到预算
> 耗尽。方法论不该有单点故障，尤其是无人值守。

- 给 `fp4_decode_kernel.py` 的 `@cute.kernel`（`:1495`）按 §6.3 的角色表加**粗粒度**
  插桩：每个角色一个 lifetime range，加 3–5 个阶段 range，**重点包住等待点**
  （pipeline acquire / mbarrier wait），因为等待占比才是判断「谁饿着谁」的依据
- 唯一名字数控制在 30 以内，单名 ≤ 32 字符
- 解决编译缓存：确认 profiling 时 kernel 确实重新 JIT，否则 trace 静默为空
- 采一条基线 trace 存档，作为后续所有轮次的对照

插桩本身**不改变 kernel 语义、也不改变干净 run 的产物**：IKET 调用在未开启 lowering
时被编译器完全剥离，不往最终 kernel 里加任何代码。这是 D1 成立的前提。

> **决策 D1（已定）**：marker 永久留在 kernel 源码里。
>
> 零成本由上述剥离机制保证，而每轮重新插桩既费时又容易插错位置，导致轮次之间的
> trace 不可比。
>
> **绑定条件：插桩后的 kernel 必须仍然过 `tests/kernel`。** 编译不过或数值变了，
> revert 插桩本身，而不是改测试。这样 D1 不会和 §6.6 第 2 条打架。
>
> 与 `CLAUDE.md` 规则 7（「不加实验性环境开关、诊断硬编码、调试 dump 路径」）的
> 张力是已知的：IKET marker 形式上属于诊断插桩。豁免理由是它更接近 NVTX 那类永久
> 标注而非临时 dump，且零开销、零语义影响。合入时在 `CLAUDE.md` 补一句显式豁免。

**验收**：
- 同一配置重跑三次，几何平均波动 < 3%。达不到就先解决计时噪声，否则后面分不清
  「改善了」和「抖动」
- `analyze_trace.py` 能在真实 decode kernel 上产出非空的角色表，且 `malformed_ranges`
  为 0
- 基线 trace 给出**当前关键路径的 warp 角色**——这是 Phase 2 的起点假设，也是第一次
  用证据而非推测来定位瓶颈
