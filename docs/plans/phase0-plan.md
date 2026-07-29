# Phase 0 Plan — 度量基建（先 IKET，后 Benchmark）

## Goal Description

在不改变 decode kernel 算法或 §12 决策的前提下，建立后续优化可依赖的两套度量基建：先完成真实 `fp4_decode_kernel.py` 上的 IKET go/no-go 与基线取证，再修正 benchmark 的覆盖、分项、基线、统计和 provenance。Phase 0 结束时必须有一个 `tests/kernel/` 全绿的提交，并留下可复现的基线证据。

开工预期：本 Phase 不以加速为目标，干净性能不应有统计显著退化；同配置三次的几何平均波动应低于 3%。IKET 若可用，应给出关键路径角色、softmax/PV 是否串行、实际 CTA/SM 分布；若两轮内不可用，按 D8 记录证据并把后续主证据切换为 ncu，不改变任何 gate。

## Acceptance Criteria

- **AC-1: 环境和执行纪律可审计。**
  - Positive Tests (expected to PASS):
    - 所有测试、benchmark、profiling 命令均显式先设置 §6.1 的 `PATH` 和 `CUDA_VISIBLE_DEVICES=1`。
    - 当前分支始终为 `perf/fp4-decode-2x`，Phase 0 总轮次不超过 10、耗时不超过 2 小时，0b 不超过 2 轮。
  - Negative Tests (expected to FAIL):
    - 使用默认 Python、其他 GPU、插桩 run 的 wall/gpu 时间作为性能结论，或修改 D0–D8。

- **AC-2: IKET marker 永久、粗粒度且不改变语义。**
  - Positive Tests (expected to PASS):
    - `@cute.kernel` 内为有效 warp 角色添加 lifetime 与关键等待/阶段 range；唯一名字少于 30、每名不超过 32 字符，range 两端在同一 warp-uniform guard 内。
    - `CLAUDE.md` 明确记录 D1 的永久 marker 豁免。
    - 未开启 IKET lowering 的干净构建继续通过 `PYTHONPATH=src python -m pytest -q tests/kernel`。
  - Negative Tests (expected to FAIL):
    - 在 host wrapper 插桩、跨 warp guard 配对 range、加入环境开关/调试 dump，或为通过测试放宽数值门槛。

- **AC-3: 真实 decode 上完成 IKET go/no-go。**
  - Positive Tests (expected to PASS):
    - profiling 路径显式清理/绕过本仓库编译缓存，确保真实 kernel 在 `run-iket` 子进程中 JIT。
    - 两轮预算内至少一条真实 decode trace 能由 `analyze_trace.py` 产生非空角色表且 `malformed_ranges == 0`；报告关键路径角色、softmax→PV 序列化证据、grid/CTA/SM 分布。
    - 若两轮均失败，记录每轮失败原因和已排除分支，并正式登记 D8 的 ncu fallback。
  - Negative Tests (expected to FAIL):
    - 空 launch/空角色表被当成“无瓶颈”，或 IKET 与 ncu 同一趟运行。

- **AC-4: Benchmark 覆盖并固化 D0/D4。**
  - Positive Tests (expected to PASS):
    - grid 覆盖 batch `{1,2,4,8,16,32,64,128}`、seqlen `{1024,4096,16384,65536}`，并含至少一组 MHA、GQA、MQA。
    - 能运行 `batch=64/128 × seqlen=65536`，默认 token 上限至少 8,400,000，且先实测峰值显存可承受。
    - FA4 始终走 varlen，并分别记录 `num_splits=1`、启发式及二者较优值；gate 基线字段明确为较优者（D0）。
    - 同时输出 BF16-Q 与 FP4-Q 路径字段（D4）；Phase 1 前 FP4-Q 可明确标为尚未实现，而不得伪造数据。
  - Negative Tests (expected to FAIL):
    - 丢弃长序列格子、只测 GQA-8、使用固定 `fa4_bf16` 代替较优者、或 split-K FA4 不走 varlen。

- **AC-5: 分项、统计与 provenance 完整。**
  - Positive Tests (expected to PASS):
    - per-kernel 数据非空并可区分 Q 量化、attention/核心 decode 和其他可见 kernel；报告 Q 量化占 BF16-Q 总 GPU 时间的比例。
    - 输出每 seqlen 的 batch 几何平均、全表几何平均和最差单点；后两者只报告不设 gate。
    - 结果记录环境路径、Python/CUTLASS/flash-attn 版本、GPU 型号、Git commit、参数、时间戳和 FA4 split 数。
    - 一个代表配置三次干净 run 的几何平均波动低于 3%。
  - Negative Tests (expected to FAIL):
    - `kernels` 为空、只输出算术平均、无 commit/env 信息，或把 profiler 时间混入干净结果。

- **AC-6: Phase 0 以绿色提交收尾。**
  - Positive Tests (expected to PASS):
    - `PYTHONPATH=src python -m pytest -q tests/kernel` 全绿。
    - Phase 0 报告记录预期、实测、失败分支、IKET go/no-go 结论和后续证据工具优先级。
    - 工作树干净，存在一个明确的 Phase 0 收尾提交。
  - Negative Tests (expected to FAIL):
    - 测试红仍继续、未记录失败分支、或只留下未提交改动。

## Path Boundaries

### Upper Bound (Maximum Scope)

- 可修改 `tests/kernel_profile/`、IKET 分析 helper、`src/nvfp4_decode_kernel/fp4_decode_kernel.py` 中仅用于永久 marker 的位置、与编译缓存清理直接相关的 profiling 辅助代码、`CLAUDE.md` 的 D1 豁免说明，以及 Phase 0 报告/计划文件。
- 可为 benchmark 可测试性增加小型单元测试或纯 Python 聚合测试。

### Lower Bound (Minimum Scope)

- 必须完成 0b go/no-go（成功 trace 或两轮有证据的 fallback）以及 0a 的完整覆盖、统计、provenance、per-kernel 分项和稳定性验证。

### Allowed Choices

- Can use: IKET 作为时序/结构主证据；IKET 两轮失败后按 D8 使用 ncu；CUDA events、Torch profiler/NVTX 相关 API用于干净 run 之外的 kernel 名称拆分；纯 Python JSON/统计测试。
- Cannot use: 修改 kernel 算法、放宽数值或性能 gate、更换 D0、引入长期实验环境开关、把插桩 run 当性能数据、把 FP4-Q 路径改成 D2 之外的接口。

## Dependencies and Sequence

### Milestones

1. **0b-1：定位 marker 与缓存边界**
   - 阅读 kernel warp 角色与关键 pipeline wait；设计少于 30 个名字的 marker 表。
   - 查明 `cute.compile` 缓存层次，建立 profiling 子进程内可复现的清缓存入口。
2. **0b-2：IKET 两轮 go/no-go**
   - 轮 1：最小真实 case，采集并分析。
   - 轮 2（仅轮 1 失败）：根据具体错误缩减事件量/调整 buffer/修正 JIT；仍失败则登记 ncu fallback。
3. **0a-1：Benchmark schema 与统计**
   - 设计带 schema/provenance 的结果；实现 D0、几何平均、最差单点、split 记录。
4. **0a-2：覆盖与分项**
   - 扩 batch/head grid，放开长序列 token 上限，修 per-kernel 分项，记录 BF16-Q 与 FP4-Q 状态。
5. **0a-3：实测与稳定性**
   - 跑代表格三次稳定性、显存 smoke 和必要基线；保存干净结果与报告。
6. **收尾**
   - 跑全量 `tests/kernel`，审阅 blackwellGPU red flags，记录失败分支，提交绿色状态。

## Implementation Notes

- 每次执行命令都必须在该 shell 顶部写出 §6.1 两行环境设置。
- 禁止调用外部 `codex exec` / `codex review`；每轮独立 review 必须新 launch
  内部 sub-agent，并把结果写入对应 `round-*-review-result.md`。其余 RLCR 工件、
  commit、修复和验证纪律不变。
- 主 review 使用 GPT-5.6-Terra/high，另用不同内部模型做交叉 review。
- 先读 trace JSON 再看 Perfetto；空 trace 必须报错。
- 计划文件中的 AC/Milestone 术语不得进入产品代码注释。
- §12 的 D0–D8 是不可变约束；任何需要改变它们的情况必须停止并等待用户。
