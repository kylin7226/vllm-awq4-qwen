# Qwen3.6-27B 文本大模型部署指南

本项目通过 vLLM 提供 OpenAI 兼容的 API，服务 Qwen 3.6-27B AWQ-INT4 量化模型，支持 DFlash 推测解码。

## 架构概览

```
┌──────────────────────────────────────────────────┐
│  docker-compose.yml                              │
│                                                    │
│  ┌──────────────────────────────────────────┐    │
│  │  vllm (8000)                             │    │
│  │  Qwen3.6-27B-AWQ-INT4                    │    │
│  │  + z-lab/Qwen3.6-27B-DFlash (drafter)   │    │
│  │  文本 LLM · 256K 上下文 · 视觉 · 工具   │    │
│  │  DFlash N=8 推测解码                     │    │
│  └──────────────────────────────────────────┘    │
│                    │                              │
│              128 GB UMA                           │
│         AMD Strix Halo (gfx1151)                  │
└──────────────────────────────────────────────────┘
```

单一服务，端口 8000。Docker 镜像 `vllm-awq4-qwen:builder`，vLLM v0.20.0 从源码构建 + 20 个补丁。

## 模型信息

| 项目 | 值 |
|---|---|
| 目标模型 | `cyankiwi/Qwen3.6-27B-AWQ-INT4` |
| 服务名称 | `Qwen3.6-27B-AWQ4` |
| 量化方案 | AWQ-INT4, W4A16, group_size 32, compressed-tensors |
| 磁盘占用 | ~14 GiB |
| 视觉塔 | 保留 BF16（未参与量化） |
| DFlash 推测器 | `z-lab/Qwen3.6-27B-DFlash` (~2B BF16, Gated) |
| 推测器磁盘 | ~3.3 GiB |

### 性能指标

| 指标 | 值 |
|---|---|
| 单流峰值解码 | 24.8 t/s (DFlash N=8) |
| 单流均值解码 | 18.5 t/s |
| 无推测基线 | 5.6 t/s |
| 提升幅度 | +340% |
| Prefill 平均 | 33-38 t/s |
| Prefill 瞬时峰值 | 100-400 t/s |
| 三流聚合峰值 | 41 t/s (~13.5 t/s/流) |

### DFlash 接受率

| 推测 token 数 N | 均值接受/轮 | 接受率 | 每流 t/s |
|---|---|---|---|
| 0 (无推测) | n/a | n/a | 5.64 |
| 1 | 1.52 | 52% | 8.95 |
| 4 | 3.20 | 64% | 17.92 |
| 8 | 5.64-6.35 | 51-67% | 19.80 (chat) / 24.80 (responses) |

## 端点

| 端点 | 用途 |
|---|---|
| `POST /v1/chat/completions` | 标准聊天，支持 thinking/视觉/工具调用 |
| `POST /v1/responses` | OpenAI Responses API，SSE 流式分离 reasoning/output |
| `POST /v1/completions` | 原始文本补全 |

## 前置准备

### 1. 硬件要求

- AMD Ryzen AI Max+ 395 "Strix Halo" 或兼容 gfx1151 iGPU
- 128 GB UMA（BIOS 中 UMA Frame Buffer Size 设为最小 2 GB）
- Linux 主机，`/dev/kfd` + `/dev/dri` 可暴露给 Docker
- ≥ 100 GB 磁盘空间

### 2. GRUB 配置

```bash
sudo nano /etc/default/grub
# 确保包含：
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash ttm.pages_limit=30408704 amdgpu.noretry=0 amdgpu.gpu_recovery=1"
sudo update-grub
sudo reboot

# 验证：
cat /sys/class/drm/card1/device/mem_info_gtt_total
# 期望值：~124554670080（≈ 116 GiB）
```

### 3. 接受 HuggingFace Gated 模型条款

DFlash 推测器是 gated model，需要先登录并接受条款：

- https://huggingface.co/z-lab/Qwen3.6-27B-DFlash

目标模型 `cyankiwi/Qwen3.6-27B-AWQ-INT4` 是公开的，无需接受。

### 4. 下载模型

```bash
export $(grep -E '^(HF_TOKEN|VLLM_HOST_MODELS_DIR)=' .env | xargs)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download cyankiwi/Qwen3.6-27B-AWQ-INT4 --cache-dir "$VLLM_HOST_MODELS_DIR/hub" &
HF_HUB_ENABLE_HF_TRANSFER=1 hf download z-lab/Qwen3.6-27B-DFlash       --cache-dir "$VLLM_HOST_MODELS_DIR/hub" &
wait
# ~14 GB 目标 + ~3.3 GB 推测器
```

### 5. 配置 .env

```bash
cp .env.template .env
nano .env
```

必填字段：

```bash
VLLM_HOST_MODELS_DIR=/absolute/path/to/hf-cache   # HF 缓存路径
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx              # HF 令牌
```

按需调整的参数（见下方推荐配置表）：

```bash
VLLM_MAX_NUM_SEQS=1          # 并发流数
VLLM_MAX_MODEL_LEN=262144    # 上下文长度（256K 原生）
VLLM_GPU_MEMORY_UTIL=0.9     # 显存利用率
VLLM_DFLASH_N=8             # 推测 token 数
```

## 构建镜像

### 首次构建

```bash
docker compose build
```

构建过程（约 25-35 分钟）：

| 步骤 | 内容 | 耗时 |
|---|---|---|
| 1-2 | 系统依赖 + TheRock ROCm 7.13 nightly | ~5 min |
| 3-4 | Python venv + PyTorch/torchaudio/triton | ~5 min |
| 5-6 | 构建工具 + Conch Triton kernels | ~3 min |
| 7 | 克隆 vLLM v0.20.0 + 应用 20 个补丁 | ~2 min |
| 8 | 编译 vLLM（MAX_JOBS=4） | ~15-20 min |
| 8b | 安装运行时依赖 | ~2 min |
| 8c | 安装音频依赖（av, soundfile, scipy） | ~1 min |

### 仅修改补丁后重新构建

如果只修改了 `scripts/patch_strix.py` 或 `scripts/vllm_profile_cache.py`，Docker 缓存会跳过上游步骤：

```bash
docker compose build
# 约 3-5 min（仅重新执行步骤 7-8b）
```

## 启动服务

### 方式一：docker compose（推荐）

**同时启动文本 LLM 和 ASR：**

```bash
docker compose up -d vllm vllm-asr
```

**仅启动文本 LLM：**

```bash
docker compose up -d vllm
```

**查看日志：**

```bash
docker logs -f vllm-awq4-qwen
```

**等待启动完成信号：**

```
INFO:     Application startup complete.
```

冷启动约 9 分钟（模型加载 ~95s + profile_run ~7min + server startup ~5s）。使用 profile 缓存后（Patch 16），后续重启约 95 秒。

**验证服务就绪：**

```bash
curl http://127.0.0.1:8000/v1/models
# 期望返回：{"object": "list", "data": [{"id": "Qwen3.6-27B-AWQ4", ...}]}
```

### 方式二：docker 手动运行

```bash
docker run -d \
  --name vllm-awq4-qwen \
  --privileged \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --ipc host \
  --shm-size 16gb \
  -p 8000:8000 \
  -v "$VLLM_HOST_MODELS_DIR":/models:ro \
  -v "$PWD/.triton-cache":/root/.triton/cache \
  -v "$PWD/.vllm-cache":/root/.cache/vllm \
  -v "$PWD/.vllm-cache/profile":/root/.cache/vllm-profile \
  -e HF_HOME=/models \
  -e HF_HUB_OFFLINE=1 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_USE_TRITON_AWQ=1 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e MIOPEN_FIND_MODE=FAST \
  -e VLLM_SKIP_MEMORY_PROFILING=1 \
  -e VLLM_PROFILE_CACHE_DIR=/root/.cache/vllm-profile \
  vllm-awq4-qwen:builder \
  vllm serve cyankiwi/Qwen3.6-27B-AWQ-INT4 \
    --host 0.0.0.0 --port 8000 \
    --served-model-name Qwen3.6-27B-AWQ4 \
    --attention-backend ROCM_ATTN \
    --mm-encoder-attn-backend TRITON_ATTN \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --enforce-eager \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 1 \
    --max-model-len 262144 \
    --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":8}'
```

**关键参数说明：**

| 参数 | 说明 |
|---|---|
| `--attention-backend ROCM_ATTN` | DFlash 推测解码需要，支持 non-causal |
| `--mm-encoder-attn-backend TRITON_ATTN` | 视觉编码器使用 Triton 注意力（TORCH_SDPA 会产生 NaN） |
| `--reasoning-parser qwen3` | 解析 `` 推理 token |
| `--tool-call-parser qwen3_coder` | 工具调用解析 |
| `--enforce-eager` | 禁用 HIP 图捕获（gfx1151 上会冻结） |
| `--gpu-memory-utilization` | UMA 池利用率上限 |
| `--max-model-len` | 最大上下文长度，256K 原生 |
| `--speculative-config` | DFlash 推测解码配置 |

**注意：** 不要传 `--quantization` 参数。vLLM 会自动从 `config.json` 检测 compressed-tensors 并路由到 AWQMarlinLinearMethod。显式传入可能导致回退到慢速的 `ops.awq_gemm` 路径。

## 推荐配置

| 配置 | MAX_NUM_SEQS | MAX_MODEL_LEN | GPU_MEMORY_UTIL | 预算上限 | 使用场景 |
|---|---|---|---|---|---|
| **单用户，最大上下文** | `1` | `262144` | `0.9` | ~115 GiB | 单聊，完整 256K |
| **3 代理多流** ⭐ | `3` | `131072` | `0.5` | ~64 GiB（实测空闲 ~50 GiB） | 3 个并发客户端，128K 每个，为 RAG/TTS 等 CPU 服务留空间 |
| **激进 3 流** | `3` | `131072` | `0.7` | ~90 GiB | 3 客户端，更大 KV 余量 |

"预算上限" = `gpu_memory_utilization × 128 GiB UMA`。这是天花板而非目标。

## 调试模式

### 前台运行 + 详细日志

```bash
docker compose up vllm  # 不加 -d，前台运行
```

### 提高日志级别

在 docker-compose.yml 的 vllm 服务中添加：

```yaml
environment:
  VLLM_LOGGING_LEVEL: DEBUG
```

> **警告**：`DEBUG` 级别会使每个 op 都格式化参数为字符串，导致推理慢 20-100 倍。仅用于调试。

### 快速测试 — 不启动容器

```bash
# 确保已安装 vLLM 和依赖
uv pip install vllm==0.20.0

# 直接启动
vllm serve cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  --attention-backend ROCM_ATTN \
  --mm-encoder-attn-backend TRITON_ATTN \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --enforce-eager
```

> 注意：主机上需要有完整的 ROCm 环境。

### 查看 GPU 使用情况

```bash
# GTT 和 VRAM 使用量
cat /sys/class/drm/card1/device/mem_info_gtt_used
cat /sys/class/drm/card1/device/mem_info_vram_used

# 进程 GPU 占用
rocm-smi
```

### 进入运行中容器调试

```bash
docker exec -it vllm-awq4-qwen bash

# 在容器内：
# 查看 vLLM 版本
python -c "import vllm; print(vllm.__version__)"

# 手动测试模型加载
python -c "
from vllm import LLM
llm = LLM(
    model='cyankiwi/Qwen3.6-27B-AWQ-INT4',
    enforce_eager=True,
    max_model_len=8192,
)
print('Model loaded successfully')
"
```

## 稳定运行模式

### 健康检查

```bash
# 基本健康检查
curl -s http://127.0.0.1:8000/health
# 期望: 200 OK

# 检查模型
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool

# 快速聊天测试
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B-AWQ4","messages":[{"role":"user","content":"hi"}]}'
```

### 停止和重启

```bash
# 仅停止 LLM（不影响 ASR）
docker compose stop vllm

# 重启 LLM
docker compose start vllm

# 完整重启（重建镜像后）
docker compose down
docker compose build
docker compose up -d vllm
```

> **重要**：`.env` 变更后需要 `docker compose down && up -d`，不能只 `restart`。`restart` 复用运行中的容器，不会重新读取环境变量。

## API 使用指南

### 基础聊天

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-27B-AWQ4",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### 带 thinking 的聊天

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-27B-AWQ4",
    "messages": [{"role": "user", "content": "解释量子纠缠"}],
    "stream": true
  }'
```

### 关闭 thinking（Responses API，推荐）

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-27B-AWQ4",
    "input": "法国的首都是什么？",
    "stream": true,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

> Patch 15 确保 `chat_template_kwargs` 通过 Responses API 流式路径正确传递。

### 带图像的聊天

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-27B-AWQ4",
    "messages": [
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        {"type": "text", "text": "描述这张图片"}
      ]}
    ]
  }'
```

### 工具调用

**推荐模式：** `/v1/chat/completions` + `stream: false`，或 `/v1/responses` + `stream: true`。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-27B-AWQ4",
    "messages": [{"role": "user", "content": "东京天气怎么样？"}],
    "stream": false,
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

### SSE 流式 Responses（带 reasoning 分离）

```python
import json
import httpx

with httpx.Client(timeout=300.0) as client:
    resp = client.post(
        "http://127.0.0.1:8000/v1/responses",
        json={
            "model": "Qwen3.6-27B-AWQ4",
            "input": "解释相对论",
            "stream": True,
        },
    )
    reasoning = ""
    output = ""
    for line in resp.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event.get("type") == "response.reasoning_text.delta":
                reasoning += event.get("delta", "")
            elif event.get("type") == "response.output_text.delta":
                output += event.get("delta", "")
    print("Reasoning:", reasoning[:200], "...")
    print("Output:", output)
```

## 故障排查

### 问题：服务启动后 /v1/chat/completions 无响应

**原因：** 可能还在 profile_run 阶段（约 7 分钟）。

**解决：** 查看日志 `docker logs -f vllm-awq4-qwen`，等待 `Application startup complete`。

### 问题：GPU 显存不足 OOM

**原因：** 显存利用率设置过高，或多流并发时 KV cache 池耗尽。

**解决：**
1. 降低 `--gpu-memory-utilization` 到 0.5-0.7
2. 减少 `--max-num-seqs`
3. 降低 `--max-model-len` 到 131072（减半 KV 预算）
4. 查看当前显存使用：`cat /sys/class/drm/card1/device/mem_info_gtt_used`

### 问题：启动慢（> 10 分钟）

**原因：** 首次启动需要 profile_run（Triton JIT 编译 + 合成推理），约 7 分钟。

**解决：**
1. 确保 `VLLM_SKIP_MEMORY_PROFILING=1` 已设置
2. 确保 `./.vllm-cache/profile/` 目录存在且可写
3. 第二次启动会读取 profile 缓存，降到 ~95 秒
4. 如果修改了 `max_model_len`、`gpu_memory_utilization` 等配置，会重新 profile（缓存 key 变了）

### 问题：DFlash 推测解码输出异常

**原因：** DFlash 上游仍在活跃开发，可能有输出质量 bug。

**解决：**
1. 先试 `num_speculative_tokens=1` 确认基线正确
2. 逐步增加到 N=4, N=8
3. 查看引擎日志中的 acceptance rate
4. 如果仍有问题，可以暂时关闭推测解码（移除 `--speculative-config`）

### 问题：工具调用在流式模式下不完整

**原因：** 上游流式工具调用解析器有 bug（vLLM PRs #40783, #40785, #40787 未合并）。

**解决：** 工具调用使用 `/v1/chat/completions` + `stream: false`，或 `/v1/responses` + `stream: true`。

### 问题：视觉输入返回乱码或 `!` 流

**原因：** 视觉编码器使用了错误的注意力后端。

**解决：** 确保 `--mm-encoder-attn-backend TRITON_ATTN` 已设置。TORCH_SDPA 在 gfx1151 上会产生 NaN/Inf。

### 问题：DFlash 工作线程在客户端断开后卡死

**症状：** API 服务器响应但新请求超时，EngineCore CPU 占用 79-200%。

**恢复：** `docker compose restart vllm-awq4-qwen`（约 9 分钟冷启动）。重启前运行 `./scripts/dump_logs.sh stuck-state` 保存诊断信息。

**缓解措施：**
- 客户端超时设置要大于预期解码时间
- 避免 `SIGKILL` 客户端，使用 `SIGINT` 或等待正常完成
- 长上下文请求（>10K prompt tokens）先用 N=1 测试
