# SM100 decode kernel 横向对比：本仓库 vs trtllm-gen vs FlashInfer FA2 vs FA4 BF16

四个 kernel 在同一批逻辑 K/V 上跑 47 个配置。前三个读 NVFP4 KV，FA4 读 BF16，是
「不量化 KV cache」的对照。

## 结论

**FA4 把机器的天花板标出来了：6.33–6.61 TB/s，且与 head 配置无关。** 四种 head
配置下 FA4 都能跑满这个数，说明这台机器的可达 DRAM 带宽在 6.6 TB/s 附近，不随
GQA group size 变化。这给了我们一把绝对的尺子——NVFP4 每 token-head 搬 144 字节，
BF16 搬 512 字节，所以一个跑满带宽的 FP4 kernel 相对 FA4 的时间比应该是
144/512 = **0.281**。按这把尺子量：

| 配置 | group | 我们峰值 TB/s | vs FA4 时间比 | 距 FP4 理想 |
|---|---|---|---|---|
| MHA 32:32 | 1 | 6.58 | 0.304 | **93%** |
| GQA 32:8 | 4 | 5.55 | 0.380 | 74% |
| GQA 32:4 | 8 | 4.13 | 0.474 | 59% |
| MQA 32:1 | 32 | 1.52 | 0.907 | **31%** |

MHA 已经基本跑满硬件（93%），MQA 只发挥了三成。**group size 的塌陷是我们自己的，
不是硬件的**——FA4 在 MQA 上照样有 6.33 TB/s。

**vs trtllm-gen**：整体几何平均 0.977（36 个可比配置，赢 20 个），总体打平。
胜负同样由 group size 决定：group 4 我们耗时是它的 84%，group 8 打平（0.955），
group 32 我们多花 16%。另外它**没有 MHA kernel**，11 个 `heads_q == heads_kv`
配置直接 `Missing TRTLLM-GEN kernel`。

**vs FlashInfer FA2**：几何平均 0.325，47 战全胜，快 1.03x 到 4.99x。

**vs FA4 BF16**：几何平均 0.476，47 个里赢 43 个，最好快 4.01x，最差慢 1.23x。
输的四个全是 MQA 长序列/高 batch——在那里 FP4 省下的 3.56x 字节完全没换成时间。

## 对比设置

四边都拿到**已经量化好**的输入（FA4 直接读原始 BF16 页），谁都不为量化付钱。
计时用 CUDA graph replay（warmup 30，20 iters × 5 repeats 取中位数），四边都不含
host dispatch。

| | 本仓库 | trtllm-gen | FlashInfer FA2 | FA4 |
|---|---|---|---|---|
| KV 精度 | NVFP4 | NVFP4 | NVFP4 | BF16 |
| Q 输入 | E2M1 FP4 | E4M3 FP8 | BF16 | BF16 |
| K/V 在 MMA 前 | E2M1 直接进 block-scaled MMA | 先反量化成 FP8 | 先反量化成 BF16 | 原生 BF16 |
| 输出 | BF16 | 仅 E4M3 FP8 | BF16 | BF16 |
| block scale 布局 | tcgen05 swizzle | V 用 4-token 交织 | K/V 都线性 | 无 |
| 页布局 | `[pages, 128, h, 64]` | HND | NHD（直读，无需转置） | 原始页表 |
| 每 token-head 字节 | 144 | 144 | 144 | 512 |

四条算术流水线不同，所以精度那一列比的是**四套不同的数值路径**，不是同一个算法的
四种实现。三个 NVFP4 边各用自己的 quantizer 从同一批 BF16 页量化。

FA4 走 varlen 入口以保住 pack-GQA，`num_splits` 取 1 与启发式两者中较快的那个
（任务文档决策 D0，最严格的读法）。**FA4 的 TB/s 是按 BF16 字节算的，不能和另外三列
直接比大小**，跨精度只有时间比有意义。

网格：`heads_q=32`，`heads_kv ∈ {32, 8, 4, 1}`，`seqlen ∈ {4096, 16384, 65536}`，
`batch ∈ {1, 4, 16, 64}`，page 128，head_dim 128。

环境：GPU 0（compute cap 10.0，183 GB），commit `beb9379`，torch 2.11.0+cu130，
flashinfer 0.6.15.post1，nvidia-cutlass-dsl 4.6.0。

## 性能

### 峰值带宽

| 配置 | group | 本仓库 | trtllm-gen | FA2 | FA4（BF16 字节） |
|---|---|---|---|---|---|
| MHA 32:32 | 1 | **6.58** | 无 kernel | 1.42 | 6.61 |
| GQA 32:8 | 4 | **5.55** | 5.13 | 1.32 | 6.53 |
| GQA 32:4 | 8 | 4.13 | **5.00** | 1.34 | 6.57 |
| MQA 32:1 | 32 | 1.52 | **2.25** | 0.43 | 6.33 |

单位 TB/s，取该配置在整个 batch × seqlen 网格上的最大值。前三列同口径（NVFP4
字节），第四列是 BF16 字节。

### 比值几何平均（我们 / 基线，小于 1 表示我们快）

| 配置 | vs trtllm-gen | vs FA2 | vs FA4 | 配置数 |
|---|---|---|---|---|
| MHA 32:32 | 不可比 | 0.229 | 0.304 | 11 |
| GQA 32:8 | **0.840** | 0.289 | 0.380 | 12 |
| GQA 32:4 | 0.955 | 0.374 | 0.474 | 12 |
| MQA 32:1 | **1.163** | 0.435 | 0.907 | 12 |
| 全部 | 0.977 | 0.325 | 0.476 | 47 |

### 主要发现：per-CTA 代价随 group size 单调塌陷，而硬件不会

| group（每 CTA 持有的 query 行数） | 1 | 4 | 8 | 32 |
|---|---|---|---|---|
| 本仓库峰值 TB/s | 6.58 | 5.55 | 4.13 | 1.52 |
| trtllm-gen 峰值 TB/s | 无 | 5.13 | 5.00 | 2.25 |
| FA4 峰值 TB/s（BF16 字节） | 6.61 | 6.53 | 6.57 | 6.33 |

FA4 那一行几乎是水平的，这就排除了「group 大的时候机器本身就跑不快」这个解释。
trtllm-gen 从 group 4 到 8 只掉 2.5%（5.13 → 5.00），我们掉 26%（5.55 → 4.13）。

这与 `docs/tasks/2.fp4_decode_speedup.md` §11 记录的「每线程代价正比于它持有的
查询行数」以及 `sm_rowmax` 是 softmax 头号开销（806 ns，`sm_exp` 的 4.5 倍）
指向同一处，现在有了四个点的完整曲线和一条硬件基准线。

一个必要的限制说明：本网格固定 `heads_q = 32`，group size 与 `heads_kv` 完全共线，
单靠这批数据分不开两者。任务文档已用 32:4 与 64:8（同为 group 8）排除了
`heads_kv`，本报告沿用该结论，没有重新验证。

### 各配置 vs FA4，按 seqlen 拆

| 配置 | 4096 | 16384 | 65536 | 全部 |
|---|---|---|---|---|
| MHA 32:32 | 0.332 | 0.296 | 0.278 | **0.304** |
| GQA 32:8 | 0.421 | 0.371 | 0.351 | **0.380** |
| GQA 32:4 | 0.486 | 0.470 | 0.466 | **0.474** |
| MQA 32:1 | 0.799 | 0.829 | 1.125 | **0.907** |

理想值 0.281。除 MQA 外，序列越长越接近理想——长序列摊薄了固定开销。MQA 反向，
到 65536 已经输给 BF16。

### 各配置 vs FA4，按 batch 拆

| 配置 | b=1 | b=4 | b=16 | b=64 |
|---|---|---|---|---|
| MHA 32:32 | 0.329 | 0.310 | 0.298 | 0.268 |
| GQA 32:8 | 0.445 | 0.425 | 0.354 | 0.310 |
| GQA 32:4 | 0.448 | 0.508 | 0.498 | 0.444 |
| MQA 32:1 | 0.847 | 0.808 | 0.987 | 1.001 |

注意 MHA b=64 的 0.268 **低于理想值 0.281**（s=16384 那一格是 0.250）。理想值假定
FA4 已经跑满，而 FA4 在高 batch 的 MHA 上并没有，所以这里我们的有效带宽反而超过了
FA4 的——不代表突破了硬件上限，只说明这一格的分母偏松。

### 对 trtllm-gen：batch 越大、序列越长，我们越吃亏

| 配置 | b=1 | b=4 | b=16 | b=64 |
|---|---|---|---|---|
| GQA 32:8 | 0.631 | 0.878 | 0.866 | 0.944 |
| GQA 32:4 | 0.675 | 0.924 | 1.145 | 1.215 |
| MQA 32:1 | 1.006 | 0.967 | 1.303 | 1.439 |

低 batch 优势明显（整体 0.746），高 batch 交出去（整体 1.184）。低 batch 下总 CTA
数远小于 SM 数，两边都靠 split-K 填机器，我们填得更好；batch 一大这个自由度消失，
剩下的就是 per-CTA 效率之差。这条趋势与 group size 趋势可能同源，但本网格没有
直接证据把两者绑在一起。

按 seqlen 看是 0.893 / 0.932 / 1.121（4096 / 16384 / 65536）。

### 长序列明细（s=65536，微秒）

| case | 本仓库 | trtllm-gen | FA2 | FA4 |
|---|---|---|---|---|
| gqa4_b1 | 22.6 | 26.8 | 67.0 | 51.5 |
| gqa4_b16 | 217.8 | 245.5 | 923.5 | 731.1 |
| gqa8_b1 | 16.6 | 20.7 | 43.2 | 36.3 |
| gqa8_b16 | 161.8 | 127.4 | 461.8 | 326.6 |
| mqa_b1 | 32.9 | 22.7 | 34.1 | **26.8** |
| mqa_b16 | 107.4 | 75.9 | 349.2 | **90.7** |
| mha_b1 | 56.2 | 无 kernel | 237.1 | 186.4 |
| mha_b16 | 810.8 | 无 kernel | 3712.0 | 2955.1 |

MQA 长序列是唯一 BF16 FA4 反超 FP4 的地方。

### FA2 的天花板是平的

FA2 在全部 47 个配置上带宽中位数 1.01 TB/s，最大 1.42 TB/s，不管 group size、
seqlen 还是 batch 怎么变都顶在 1.3–1.4 附近。这不是带宽受限，是反量化加
`mma.sync` 的计算受限：它拿到了 FP4 的容量收益（KV 池 1.78x），完全没拿到
Blackwell 的算力收益，连 BF16 的 FA4 都跑不过。

## 数值

对 BF16 全精度参考取 cosine，batch ∈ {1,4,16} × 全部 seqlen 与 head 配置：

| 配置 | 本仓库 | trtllm-gen | FA2 | FA4 |
|---|---|---|---|---|
| MHA 32:32 | 0.9873–0.9879 | 无 kernel | 0.9953–0.9955 | 1.0000 |
| GQA 32:8 | 0.9861–0.9882 | 0.9419–0.9513 | 0.9951–0.9957 | 1.0000 |
| GQA 32:4 | 0.9869–0.9877 | 0.9440–0.9484 | 0.9951–0.9956 | 1.0000 |
| MQA 32:1 | 0.9830–0.9895 | 0.6635–0.6734 | 0.9944–0.9961 | 1.0000 |

FA4 全程 1.0000，它读的就是参考用的那批 BF16 页，这一列的作用是确认参考实现本身
没问题。剩下三个的排序符合各自的流水线设计：FA2 反量化到 BF16 后全程 BF16 算，
最准；我们把 E2M1 直接喂给 block-scaled MMA，0.987 左右且跨全部配置极稳；
trtllm-gen 多一次 FP4→FP8 的有损反量化，Q 也是 FP8，输出还只能是 FP8。

### 顺带修掉的一个 benchmark bug

原来的 `bench_vs_trtllm.py` 用 `NVFP4_GLOBAL_RANGE = 6 * 448` 算全局 descale，
这让 per-block 的 E4M3 scale 顶到 448。但 trtllm-gen 在 MMA 前要把 FP4 反量化成
**FP8**，所以 `e2m1 * block_scale` 必须落在 E4M3 内——E2M1 最大值是 6，block scale
就必须 ≤ 448/6 ≈ 74.7。原来的取值让最大的那些 K/V 项在 kernel 内部静默截断。

改成 `NVFP4_GLOBAL_RANGE = 448`（与 FlashInfer 自己的
`nvfp4_quantize_paged_kv_cache` 一致）后，`gqa4_b4_s4096` 的输出模长比从 **0.653
回到 1.036**，cosine 从 0.925 升到 0.946。上表是修正后的数据。

### 两处仍未解释的现象

**trtllm-gen 的 MQA cosine 稳定在 0.66**，跨 batch 和 seqlen 都一致，而同样配置下
我们 0.986、FA2 0.995、FA4 1.000。这不是输出缩放的问题：把 output descale 从保守的
`amax(V)` 界一路收紧到贴合输出，cosine 反而从 0.665 掉到 0.597。原因未查明，
**不应据此判定 trtllm-gen 的 MQA 质量**——更可能是本 harness 驱动它的方式在
group 32 下有问题。

**trtllm-gen 的输出不随 `bmm2_scale` 线性放大**。在 `gqa4_b4_s4096` 上把
out_descale 收紧到只用 25%/50%/80%/100% 的 E4M3 量程，实际 FP8 输出最大值分别是
88/88/96/96，线性预期应该是 112/224/358/448。超过约 5 倍之后就不再跟随。因此
harness 保留了保守的 `amax(V)` 界——它虽然浪费了 99% 的量程，却是实测最准的那个。

## 这对后续工作意味着什么

1. **group > 4 的塌陷现在有了绝对标尺，是最值得投的一项。** FA4 证明机器在任何
   group 下都能给 6.3–6.6 TB/s，所以 group 8 的 4.13 和 MQA 的 1.52 全是我们自己
   的损失，分别只发挥了 59% 和 31%。group 8 又是最常见的生产配置。下一步按任务
   文档 §11 第 1 项，用 `probe_softmax_ceiling.py` 在 group 8 上取上限，区分
   softmax 与搬运。
2. **MQA 长序列已经被 BF16 FA4 反超**（s=65536 上 b=1/16/64 三档都输，只有 b=4
   以 0.945 险胜）。也就是说在这个形态下量化 KV cache 目前是净亏——省了显存，
   赔了时间。这是比「不如 trtllm-gen」更硬的一条结论。
3. **MHA 是一块无人竞争、且已接近打满的地盘**：trtllm-gen 没有 kernel，FA2 只有
   1.42 TB/s，我们 6.58 TB/s 是 FP4 理想值的 93%，这条线上没什么可挤了。
4. **高 batch 的劣势与 group 塌陷可能同源**，都是 per-CTA 效率问题，值得先假设是
   一个修法。
5. **FA2 适合当数值回归的第二参照**：BF16 进 BF16 出，没有 FP8 staging 也没有输出
   缩放这一层，跨全部形状（含 MHA、MQA）都能跑，cosine 稳定 0.995。

## 复现

```bash
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:tests/kernel_profile

# 性能网格（47 配置，约 100 秒）
python tests/kernel_profile/bench_vs_trtllm.py --device 0 \
  --head-configs 32:8,32:4,32:1,32:32 \
  --seqlens 4096,16384,65536 --batches 1,4,16,64 \
  --iters 20 --warmup 30 --repeats 5 --with-fa4 \
  --out docs/reports/data/baseline-perf.json

# 数值网格（--check 会构造 BF16 全精度参考，batch 不能太大）
python tests/kernel_profile/bench_vs_trtllm.py --device 0 \
  --head-configs 32:8,32:4,32:1,32:32 \
  --seqlens 4096,16384,65536 --batches 1,4,16 \
  --iters 10 --warmup 10 --repeats 3 --with-fa4 --check \
  --out docs/reports/data/baseline-numerics.json
```

不加 `--with-fa4` 就只跑三个 NVFP4 kernel，快一半。原始数据在
`docs/reports/data/`，每行含四边的微秒数、TB/s、比值、FA4 选中的 `num_splits`，
以及某一边没有 kernel 时的报错原文。
