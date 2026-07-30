# Track C — host 派发开销，以及它如何污染了整个 gate 读数

分支 `perf/fp4-decode-2x`，commit `2257ab8`，GPU 1（B200，锁频 1965，负载下
1845 MHz）。脚本 `tests/kernel_profile/probe_host_dispatch.py` 与
`tests/kernel_profile/probe_graph_gate.py`，原始数据
`docs/perf/phase7/host-dispatch.json` 与 `docs/perf/phase7/graph-gate.json`。
本 Track 不改 `src/`。

## 结论先说

**`§8.1` 的 gate 一直在用 CUDA event 计 eager 模式的时间，而这个时间里有相当一部分
是 Python 派发，不是 kernel。** 两边付的 Python 代价还不一样，也不随形状同向变化，
所以 event 比值既不是 kernel 的比值，也不是任何稳定量的比值。

把两边都放进 CUDA graph 重测（decode 在真实服务里本来就跑在 graph 里），每个 seqlen
在 batch 上的几何平均变成：

| seqlen | event 比值（旧读数） | graph 比值（真实） | 变化 |
|---:|---:|---:|:--|
| 1024 | 1.426 | **2.618** | 大幅变差 |
| 4096 | 1.851 | **1.352** | 变好 |
| 16384 | 1.451 | **1.192** | 变好 |
| 65536 | 1.293 | **1.290** | 不变 |

gate 是 0.5。所以真实的差距是：长序列还差 2.4x 到 2.7x，短序列差 5.2x。

**这条修正把 Track A 的优先级顶到了第一位。** 按旧读数，s1024 是四个 seqlen 里
最接近达标的一个；按真实读数，它是唯一一个差出 2 倍以上的，比其他三个加起来还难。

单点数据，`--iters 20 --repeats 5`：

| 形状 | fp4 graph | fa4 graph | 比值 | event 比值 |
|:--|---:|---:|---:|---:|
| b1 s1024 | 42.4 us | 12.5 us | 3.397 | 1.330 |
| b4 s1024 | 43.1 us | 14.4 us | 2.986 | 1.321 |
| b16 s1024 | 44.1 us | 17.4 us | 2.530 | 1.290 |
| b64 s1024 | 117.3 us | 64.1 us | 1.830 | 1.823 |
| b1 s4096 | 22.7 us | 14.8 us | 1.537 | 2.034 |
| b4 s4096 | 22.7 us | 22.7 us | 1.000 | 1.922 |
| b16 s4096 | 67.7 us | 49.1 us | 1.379 | 1.905 |
| b64 s4096 | 291.1 us | 184.4 us | 1.578 | 1.578 |
| b1 s16384 | 22.7 us | 24.3 us | **0.931** | 1.558 |
| b4 s16384 | 60.5 us | 53.4 us | 1.132 | 1.515 |
| b16 s16384 | 220.3 us | 165.0 us | 1.336 | 1.362 |
| b64 s16384 | 1077.9 us | 752.3 us | 1.433 | 1.380 |
| b1 s65536 | 71.9 us | 53.4 us | 1.346 | 1.533 |
| b4 s65536 | 213.3 us | 168.4 us | 1.266 | 1.284 |
| b16 s65536 | 827.3 us | 692.9 us | 1.194 | 1.086 |
| b64 s65536 | 4174.9 us | 3069.7 us | 1.360 | 1.310 |

`b1 s16384` 已经比 FA4 快，这是全表第一个。旧读数把它记成 1.558x 慢。

## 为什么 event 读数不可信

把 wall、event、kernel-only、graph replay 四个量并排放，b1 s16384：

| | wall | event | kernel 之和 | graph replay | host 空档 |
|:--|---:|---:|---:|---:|---:|
| fp4 | 99.9 us | 99.0 us | 24.7 us | 22.7 us | 75.2 us |
| fa4 | 65.1 us | 64.5 us | 21.9 us | 24.7 us | 43.2 us |

GPU 只干了 25 us 的活，event 却读到 99 us。差出来的 75 us 是 CPU 在发起下一次
launch 之前的 Python，GPU 在这段时间里空转。CUDA event 记的是 GPU 时间轴上的跨度，
空转也算在内，所以 event 读数在 GPU 时间低于 host 时间的形状上量的其实是 Python。

wall 与 event 几乎相等（99.9 对 99.0），这是 host 受限的判据：队列从来没有积压，
CPU 一直是瓶颈。反过来 b32 s16384 的 host 空档只有 2.1 us，因为 GPU 那 557 us 足够
把 Python 全部盖住。分界线在 GPU 时间约 100 us：低于它的形状 event 读数都不可信，
这覆盖了 s1024 的全部、s4096 的低 batch、s16384 的 batch ≤ 4。

两边的 host 代价不对称，这才是问题所在。FA4 大致恒定付 43 到 55 us。FP4 付多少取决于
它走哪条内部路径：

- b1 **s1024**，`split_k_heuristic` 给出 `num_splits=1`，走单 kernel 路径，host 空档
  只有 **12.3 us**。
- b1 **s16384**，走 split 路径，host 空档 **75.2 us**。

同一个 kernel，同一个 batch，仅仅因为 split 启发式选了不同的分支，Python 代价就差了
63 us。在 s1024 上 FP4 的 Python 比 FA4 少 43 us，于是 event 比值（1.330）比真实的
graph 比值（3.397）好看了 2.6 倍；在 s16384 上 FP4 的 Python 比 FA4 多 32 us，于是
event 比值（1.558）比真实的（0.931）难看了 1.7 倍。**旧读数在短序列上高估了我们，在
长序列低 batch 上低估了我们**，方向相反，所以也不能用一个统一的修正系数补回去。

## 那 63 us 花在哪

`cProfile`，b1 s16384，split 路径，每次调用的自身耗时（cProfile 自身有约 1.9 倍的
放大，看相对占比而非绝对值）：

| 自身 us | 累计 us | 次数 | 位置 |
|---:|---:|---:|:--|
| 19.7 | 112.8 | 1 | `_decode.py:905 decode_fp4_split` |
| 12.3 | 41.2 | 1 | `_decode.py:475 decode_fp4`（**纯校验，结果丢弃**） |
| 10.0 | 10.4 | 1 | CuTe DSL `<string>:2 wrapper` |
| 9.7 | 9.7 | 3 | `torch.empty` |
| 7.7 | 173.0 | 1 | `_kernel.py:8 fp4_decode_impl` |
| 6.7 | 11.4 | 4 | `_decode.py:108 _page_stride_bytes` |
| 6.6 | 6.6 | 28 | `Tensor.stride` |
| 6.3 | 13.0 | 4 | `torch/_utils.py:517 _get_device_index` |
| 5.6 | 5.6 | 4 | `Tensor.as_strided` |
| 5.5 | 34.1 | 1 | `split_k_combine.py:694 flash_attn_combine` |
| 4.8 | 12.0 | 4 | `_decode.py:136 _page_scales_for_kernel` |

没有单一热点，是一堆每次调用都重算一遍的常量。四个最大的块：

1. **丢弃的校验（累计 41 us）。** `_kernel.py:113` 在 split 路径上先调
   `decode_fp4(..., validate_only=True)` 走一遍完整合同检查，再调
   `decode_fp4_split`，后者又自己重做了一部分（`_decode.py:939` 的 8 个
   `_require_cuda_tensor`）。注释说这是为了「保持单一公共入口的完整合同检查」，目的
   正当，但代价是每步解码重跑一次几百行 Python 形状检查，而这些检查的结果在同一个
   序列的所有解码步之间是不变的。
2. **`flash_attn_combine` 作为独立 Python 入口（累计 34 us）。** 它有自己的一套校验
   与分配，与主 launch 完全分离。
3. **重算张量元数据（`_page_stride_bytes` + `_page_scales_for_kernel` + 28 次
   `stride()` + 4 次 `as_strided`，累计约 35 us）。** 这些只取决于形状与 stride，在
   解码循环里逐步不变。
4. **3 次 `torch.empty`（10 us）。** split 的 `output_partial` / `lse_partial` 工作区
   每次重新分配。

顺带确认一个不成立的猜想：`_check_device_values`（`_decode.py:184`，内含
`torch.any(...).item()` 这个设备同步）**不在测量路径上**，因为
`tests/kernel_profile/bench_decode.py` 全部传了 `trusted_metadata=True`。设备同步不是
本次 host 开销的来源。但要注意，任何**不**传 `trusted_metadata=True` 的调用方每步都会
吃一次设备同步，那会比这里的 90 us 更糟。

## 建议

**第一，把 gate 的计时改成 graph replay。** 这既是更诚实的 kernel 测量，也更贴近
decode 的真实运行方式。`probe_graph_gate.py` 里的 `graph_us()` 可以直接搬进
`bench_decode.py`。改之前要先把 §8.1 与 D0/D4 里「CUDA event」的措辞一并更新，并把
上表作为新的基线记录在案——这是对 D0 基线定义的实质修改，按 D7 属于**人的决定，
agent 不得自行改**。本文只提出建议。

**第二，host 开销本身要不要修，取决于第一条怎么定。** 如果 gate 走 graph，那 90 us
的 Python 完全不影响达标，可以降到最低优先级；它只影响不用 graph 的调用方。如果
gate 仍然走 eager，那它就是 s4096 及以下所有形状的首要瓶颈，比 softmax 还大，按代价
排序的修法是：

| 修法 | 预计收回 | 代价 | 位置 |
|:--|---:|:--|:--|
| split 路径去掉丢弃的那次完整校验，把合同检查下沉到 `decode_fp4_split` | ~22 us | 低 | `_kernel.py:113` |
| 缓存张量元数据（stride、`as_strided` 视图）到 compile cache 条目上 | ~18 us | 中 | `_decode.py:108/136` |
| combine 复用常驻工作区，不每次 `torch.empty` | ~5 us | 低 | `_decode.py:950/959` |
| combine 的 Python 校验与主 launch 合并 | ~18 us | 中 | `split_k_combine.py:694` |

## 与另外两条 Track 的关系

Track A 查的 s1024 那 28 us 地板是 kernel-only 量出来的，**不受本文影响，仍然成立**。
本文只是说明它比原先以为的更要紧：s1024 的真实比值是 2.618，是四个 seqlen 里唯一
差出 2 倍以上的。

Track B 的带宽测量用的也是 kernel-only，同样不受影响。但它今后的读数建议改用 graph
replay，理由同上，而且 graph 让 launch 间隔也进入测量，对多 kernel 的 split 路径更
公平。
