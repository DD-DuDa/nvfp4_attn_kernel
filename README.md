# NVFP4 Decode Kernel

SM100 上的分页 NVFP4 解码注意力内核（`src/nvfp4_decode_kernel/`），以及把它接进
vLLM v1 的注意力后端（`src/nvfp4_vllm/`）。

- 内核契约、张量布局、开发规则见 [`CLAUDE.md`](CLAUDE.md)。
- vLLM 集成的设计与实测记录见 [`docs/tasks/1.vllm_v1_design.md`](docs/tasks/1.vllm_v1_design.md)。

## 硬件

**只支持 SM100**（compute capability 10.0）。内核在别的架构上编译不出来，所有需要它
的测试会 skip 而不是失败——所以在错的卡上跑，套件是"绿"的，但什么也没验证。本机 8 卡。

## Python 环境

**必须用 `../BitKV_nvfp4/_local/envs/vllm-nvfp4`。PATH 上的 `python` 不行。**

两个原因，各自都是硬性的：

1. **vLLM 只在那棵树里编译过。** `_C_stable_libtorch.abi3.so` 一整套都在
   `BitKV_nvfp4/third_party/vllm` 下；本仓库的 `third_party/vllm` submodule 一个
   `.so` 都没有。
2. **默认环境的 CuTeDSL 没有 `iket`。** 本机 PATH 上的 `python` 指向
   `/opt/conda/envs/torch-base`，它的 `nvidia-cutlass-dsl` 缺 `iket`，于是
   `import nvfp4_decode_kernel` 直接炸：

   ```
   ImportError: cannot import name 'iket' from 'cutlass.cute.experimental'
   ```

   内核里的 IKET range 是常驻的（`CLAUDE.md` 开发规则第 7 条），不是可选依赖。

环境内容（Python 3.12.13，uv venv）：

| 包 | 版本 |
| --- | --- |
| torch | 2.11.0+cu130 |
| vllm | 0.1.dev1+g1ad84fea8.precompiled（editable） |
| nvidia-cutlass-dsl | 4.6.0 |
| cuda-python | 13.0.3 |
| apache-tvm-ffi | 0.1.10 |
| torch-c-dlpack-ext | 0.1.5 |
| flash-attn-4 | 4.0.0b23（仅测试用） |
| flashinfer-python | 0.6.15.post1（仅测试用） |
| datasets | 5.0.0（仅 GSM8K 用） |
| pytest | 9.1.1 |

本仓库以 editable 方式装在里面（`__editable__.nvfp4_attn_kernel-0.1.0.pth`），所以
`import nvfp4_decode_kernel` / `import nvfp4_vllm` 直接指向 `src/`，改代码不用重装。

`pyproject.toml` 的 entry point 把 vLLM 的 `backend: "CUSTOM"` 指向本仓库的后端，
这也是靠 editable 安装生效的——环境里没装这个包，`attention_config={"backend":
"CUSTOM"}` 就找不到后端。

> **注意**：`import vllm` 解析到的是 `BitKV_nvfp4/third_party/vllm`，**不是**本仓库的
> submodule。这么用的前提是两棵树在 nvfp4 相关的五个文件上逐字节相同
> （`v1/kv_cache_interface.py`、`utils/torch_utils.py`、`v1/attention/backend.py`、
> `v1/attention/backends/registry.py`、
> `model_executor/layers/attention/attention.py`）。**将来两棵树在这些文件上分叉，
> 这个前提立刻失效，必须重新对齐**，否则测的就不是要发布的那份代码。

## 模型与数据

e2e 套件要一个真实模型。默认下载 `NousResearch/Meta-Llama-3.1-8B-Instruct`，但本机
已有本地副本，用 `NVFP4_TEST_MODEL` 指过去，避免把机器绝对路径写进仓库：

```
dev/models/Meta-Llama-3.1-8B-Instruct
```

`test_vllm_integration.py` 另外要 GSM8K，已在 `~/.cache/huggingface` 里。

## 跑测试

```bash
scripts/run_tests.sh kernel      # 内核数值与量化
scripts/run_tests.sh e2e         # vLLM 集成（不含 soak）
scripts/run_tests.sh soak        # 多请求下的 slot 复用
scripts/run_tests.sh all

scripts/run_tests.sh kernel -- -k residual -x     # -- 之后原样转给 pytest
```

脚本会挑对解释器、把模型路径和 e2e 开关设好，然后从仓库根目录调 pytest。

| 套件 | 覆盖 | 起引擎 | 大致耗时 |
| --- | --- | --- | --- |
| `kernel` | 解码内核对 torch oracle 与 FlashAttention 的数值、Q/KV 量化的字节一致性 | 否 | ~1.5 min |
| `e2e` | 护栏、控制面、写路径、读路径、显存核算、后端透传 | 8 次 | ~4 min |
| `soak` | 一个引擎连续服务变长请求，盯非有限输出 | 1 次 | ~1.5 min |

e2e 的耗时**几乎全是引擎构造**（每次约 30 秒），不是断言本身。想缩短就减引擎数量，
减断言没用。

不设 `NVFP4_RUN_VLLM_E2E=1` 时，`tests/e2e` 里不需要引擎的部分照样跑（护栏矩阵、
控制面内核、布局、tail 重置），几十秒，只碰 GPU 不碰模型。

### 可调项

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `NVFP4_ENV` | `../BitKV_nvfp4/_local/envs/vllm-nvfp4` | 解释器所在环境 |
| `NVFP4_TEST_MODEL` | `../models/Meta-Llama-3.1-8B-Instruct` | e2e 用哪个模型 |
| `NVFP4_RUN_VLLM_E2E` | 脚本按套件自动设 | 设成 `1` 才跑要引擎的测试 |
| `NVFP4_SOAK_ROUNDS` | `6` | soak 每轮发多少批请求 |
| `NVFP4_GSM8K_N` | `32` | GSM8K 取几条 |
| `NVFP4_GSM8K_MAX_NUM_SEQS` | `8` | GSM8K 那两个引擎的并发数 |
| `CUDA_VISIBLE_DEVICES` | `0` | 用哪张卡 |

### 八卡并行灌 soak

追非确定性缺陷时，样本量靠横向铺开来。每张卡一个独立进程，各自的轮数用环境变量放大：

```bash
for gpu in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$gpu NVFP4_SOAK_ROUNDS=60 \
    scripts/run_tests.sh soak > /tmp/soak-$gpu.log 2>&1 &
done
wait
grep -l -E "FAILED|Error" /tmp/soak-*.log || echo "八卡全过"
```

一次教训写在这里免得重蹈：**负载的形状比样本量重要。** 1200 次单请求运行零故障，
也抵不过一次多请求、变长度、slot 反复易手的运行——当初就是这么把主导缺陷放过去的
（`docs/tasks/1.vllm_v1_design.md` 的 C3/C4 两节）。

### 不用脚本

```bash
ENV=../BitKV_nvfp4/_local/envs/vllm-nvfp4
CUDA_VISIBLE_DEVICES=0 $ENV/bin/python -m pytest tests/kernel -q
```

`pyproject.toml` 已经设了 `pythonpath = ["src"]`，所以不用另外给 `PYTHONPATH`，
前提是从仓库根目录启动 pytest。
