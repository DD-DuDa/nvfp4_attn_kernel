# split 启发式的护栏是用被污染的读数标定的

分支 `perf/fp4-decode-2x`，commit `2257ab8`，GPU 1（B200，148 SM，锁频 1965 /
负载 1845 MHz）。脚本 `tests/kernel_profile/probe_split_sweep.py` 与
`tests/kernel_profile/probe_graph_gate.py`，数据 `docs/perf/phase7/split-sweep.json`、
`graph-gate.json`、`graph-gate-minpages1.json`。本文不改 `src/`，只给证据。

这份结果与 Track A 的委托范围重叠。Track A 由并行 agent 独立调查，本文是从 Track C
（host 开销）顺出来的旁支，两者应当互相印证而不是互相替代。

## 结论

`split_k_heuristic` 里的 `min_pages_per_split = 8`（`_decode.py:59`）在短上下文上把
正确答案一票否决了。它的标定依据写在注释里——「at one block per split a 1K context
lost 1.7x to the combine overhead alone」——而那个 1.7x 是用 eager 模式 CUDA event
量的，其中 combine 那条路径带来的是约 34 us 的 **Python 派发**，GPU 上的 combine
kernel 只有 3.7 us。**护栏挡住的是一个 host 开销，代价却是 GPU 上的占用率。**

强制指定 split 数、用 CUDA graph replay 计时（把 host 派发从两边都排除）：

| 形状 | pages/row | 启发式选 | 该选择耗时 | 实测最优 | 最优耗时 | 提升 |
|:--|---:|---:|---:|---:|---:|---:|
| b1 s1024 | 8 | 1 | 42.1 us | **8** | **10.4 us** | 4.05x |
| b4 s1024 | 8 | 1 | 43.3 us | **4** | **12.5 us** | 3.46x |
| b16 s1024 | 8 | 1 | 43.8 us | **2** | **28.8 us** | 1.52x |
| b1 s4096 | 32 | 4 | 22.7 us | **16** | **12.6 us** | 1.80x |

完整扫描（vs FA4 的 graph 时间）：

```
b1 s1024   fa4 12.4    splits 1: 42.1 (3.400x)  2: 16.5 (1.330x)  4: 12.5 (1.006x)  8: 10.4 (0.843x)
b4 s1024   fa4 14.4    splits 1: 43.3 (2.996x)  2: 16.6 (1.147x)  4: 12.5 (0.868x)  8: 18.7 (1.294x)
b16 s1024  fa4 18.4    splits 1: 43.8 (2.389x)  2: 28.8 (1.570x)  4: 41.2 (2.245x)  8: 59.5 (3.243x)
b1 s4096   fa4 16.1    splits 1: 90.4 (5.628x)  2: 35.0 (2.181x)  4: 22.7 (1.415x)
                              8: 16.6 (1.031x) 16: 12.6 (0.782x) 32: 20.1 (1.250x)
```

`b1 s1024` 的最优是 **每个 split 只分到 1 个 page block**，正是护栏明令禁止的配置，
而它比 FA4 还快 1.19 倍。

## 「28 us 地板」不是地板，是占用率

Phase 7 的上限探针在 `s1024` 上量到：production 41 us，softmax 全删仍有 28 us，于是
记为「与 softmax 无关的固定开销」。那份测量是在 `splits=1` 下做的。

`splits=1` 时 `b1 s1024` 的 grid 是 `rows × heads_kv = 1 × 8 = 8` 个 CTA，跑在 148 个
SM 上，**占用率 5.4%**。把 split 开到 8，同一个 production kernel（softmax 一点没
省）跑 10.4 us——比「softmax 完全免费」的 27.6 us 还快 2.65 倍。

所以那 28 us 里绝大部分不是任何固定成本，是 8 个 CTA 串行做完了本可以由 64 个 CTA
并行做完的活。这也解释了「batch 1 到 16 耗时恒为 41 us」这个当时看不懂的形状：这个
区间里 CTA 数从 8 涨到 128，始终没超过 148 个 SM，所以每个 CTA 的串行工作量不变，
总时间自然不变。到 b64 时 CTA 数 512 超过 SM 数，时间才开始随 batch 线性增长。

## 护栏之外，启发式本身是对的

`target = max(2, ceil(sms / unsplit_ctas))` 这一行在四个测试点上都命中了实测最优：

| 形状 | unsplit CTA | target | 去掉护栏后选 | 实测最优 |
|:--|---:|---:|---:|---:|
| b1 s1024 | 8 | 19 | 8（被 pages=8 截断） | 8 ✓ |
| b4 s1024 | 32 | 5 | 4 | 4 ✓ |
| b16 s1024 | 128 | 2 | 2 | 2 ✓ |
| b1 s4096 | 8 | 19 | 16 | 16 ✓ |

把 `min_pages_per_split` 从 8 降到 1 时，`splits * 1 <= max_pages_per_row` 退化成
「split 数不超过 page 数」这个本来就必须成立的约束，**四个点全部命中最优**。需要改的
只有那一个常数，占用率那部分逻辑不用动。

## 全网格影响，以及两种读数的冲突

`min_pages_per_split` 8 → 1，每个 seqlen 在 batch 上的几何平均：

| seqlen | graph 8 | graph 1 | 变化 | event 8 | event 1 | 变化 |
|---:|---:|---:|:--|---:|---:|:--|
| 1024 | 2.618 | **1.227** | 好 2.13x | 1.426 | 2.068 | **变差** |
| 4096 | 1.352 | **1.134** | 好 1.19x | 1.851 | 1.848 | 持平 |
| 16384 | 1.192 | 1.213 | 差 1.8% | 1.451 | 1.438 | 持平 |
| 65536 | 1.290 | 1.326 | 差 2.8% | 1.293 | 1.309 | 持平 |

**两种读数在 s1024 上给出相反的结论**，这正是 D9 需要人拍板的原因。原因不难理解：
开 split 会多一次 combine launch，GPU 上只值 3.7 us，Python 上却值 34 us。在 graph
里那 34 us 不存在，在 eager 里它是主导项。

长序列上那 1.8% 与 2.8% 的退化不是启发式变了——`b1 s65536` 在两种护栏下都选 16
split——而是同配置的测量抖动（71.9 us 对 81.3 us，13%）。**这说明本轮 graph 计时在
低 batch 长序列上的噪声底比想象的高**，凡是小于 5% 的差异都不应当据此下结论；后续要
么加大 `--repeats`，要么在这些点上单独复测。

## 建议的动作顺序

冲突可以消掉，不必二选一。**先修 host 开销，再放松护栏**，两种读数就会同向：

1. **去掉 split 路径上那次结果被丢弃的完整校验**（`_kernel.py:113`，约 22 us
   Python），把合同检查下沉到 `decode_fp4_split`。这一步单独看就是纯收益，不改变任何
   GPU 行为。
2. **把 combine 的 Python 校验与工作区分配并进主 launch**（约 23 us）。
3. 完成 1、2 之后，split 路径的 host 代价降到与非 split 路径相当，护栏存在的理由消失，
   再把 `min_pages_per_split` 降到 1，两种读数都会改善。
4. 每一步都要跑 `PYTHONPATH=src python -m pytest -q tests/kernel`。split 结果与非
   split 结果的余弦在 0.9992 到 0.9996 之间（见 `split-sweep.json`），这是每个 split
   独立做 softmax 缩放与 P 量化的正常后果，但把 split 开到短上下文会让**更多**形状走
   上这条路径，数值门槛必须重新确认，不能只看性能。

在 D9 定下来之前不动 `src/`。以上仅为建议。
