# Qwen3-ASR 语音识别部署指南

Qwen3-ASR 是 Qwen 系列的纯语音识别模型（~8B 参数，17 种语言）。本项目通过 vLLM 提供 OpenAI 兼容的 API，支持三种调用模式。

## 架构概览

```
┌──────────────────────────────────────────────────┐
│  docker-compose.yml                              │
│                                                    │
│  ┌──────────────────┐   ┌──────────────────┐     │
│  │  vllm (8000)     │   │  vllm-asr (8001) │     │
│  │  Qwen3.6-27B     │   │  Qwen3-ASR-8B    │     │
│  │  文本 LLM        │   │  语音识别        │     │
│  │  DFlash + vision │   │  3 种模式        │     │
│  └──────────────────┘   └──────────────────┘     │
│         │                      │                  │
│         └──────────┬───────────┘                  │
│                    │ 128 GB UMA                   │
│             AMD Strix Halo (gfx1151)              │
└──────────────────────────────────────────────────┘
```

两个服务共享同一 Docker 镜像（`vllm-awq4-qwen:builder`），但使用不同的模型和端口，各自独立进程。vLLM 不支持单实例同时加载文本 LLM 和 ASR 模型。

### 三种调用模式

| 模式 | 端点 | 协议 | 输入格式 | 适用场景 |
|---|---|---|---|---|
| 非流式 | `POST /v1/audio/transcriptions` | HTTP JSON | 音频文件 | 一次性转录，等全部结果返回 |
| SSE 流式 | `POST /v1/audio/transcriptions` + `stream=true` | HTTP SSE | 音频文件 | 长音频边处理边出文字 |
| WebSocket 实时 | `ws://host:8001/v1/realtime` | WebSocket JSON | PCM16 音频块 | 麦克风实时说话，边说边出文字 |

## 前置准备

### 1. 硬件要求

- AMD Strix Halo (gfx1151) 或兼容 RDNA 3.5 iGPU
- 128 GB UMA（BIOS 中 UMA Frame Buffer Size 设为最小 2 GB）
- Linux 主机，`/dev/kfd` + `/dev/dri` 可暴露给 Docker
- ≥ 20 GB 磁盘空间（ASR 模型约 16 GB BF16）

### 2. GRUB 配置（如尚未配置）

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

### 3. 下载 ASR 模型

```bash
export $(grep -E '^(HF_TOKEN|VLLM_HOST_MODELS_DIR)=' .env | xargs)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-ASR-8B --cache-dir "$VLLM_HOST_MODELS_DIR/hub"
```

> **注意**：Qwen3-ASR-8B 是 HuggingFace 上的 gated model，需要先登录 HF 并接受模型条款。

### 4. 配置 .env

```bash
cp .env.template .env
nano .env
```

确保以下字段已填写：

```bash
VLLM_HOST_MODELS_DIR=/absolute/path/to/hf-cache   # 你的 HF 缓存路径
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx              # HF 令牌

# ASR 相关（按需修改）
VLLM_ASR_MODEL_ID=Qwen/Qwen3-ASR-8B               # 模型 ID
VLLM_ASR_HOST_PORT=8001                           # 暴露端口
VLLM_ASR_GPU_MEMORY_UTIL=0.9                      # 显存利用率
VLLM_ASR_MAX_MODEL_LEN=8192                       # 上下文长度（30s 音频足够）
```

## 构建镜像

### 首次构建

```bash
docker compose build
```

构建过程（约 30-40 分钟）：

| 步骤 | 内容 | 耗时 |
|---|---|---|
| 1-2 | 系统依赖 + TheRock ROCm 7.13 nightly | ~5 min |
| 3-4 | Python venv + PyTorch/torchaudio/triton | ~5 min |
| 5-6 | 构建工具 + Conch Triton kernels | ~3 min |
| 7 | 克隆 vLLM v0.20.0 + 应用 20 个补丁 | ~2 min |
| 8 | 编译 vLLM（MAX_JOBS=4） | ~15-20 min |
| 8b | 安装运行时依赖（含 av, soundfile, scipy） | ~2 min |
| 8c | 安装音频依赖（PyAV bundles FFmpeg） | ~1 min |

### 仅修改补丁后重新构建

如果只修改了 `scripts/patch_strix.py` 或 `scripts/vllm_profile_cache.py`，Docker 缓存会跳过 PyTorch 编译等上游步骤，只需重新执行 Patch 和应用步骤：

```bash
docker compose build
# 约 3-5 min（仅重新执行步骤 7-8b）
```

### 仅修改音频依赖

如果只添加了 `av`、`soundfile`、`scipy`（Step 8c），构建时间约 1 分钟。

## 启动服务

### 方式一：docker compose（推荐）

**同时启动文本 LLM 和 ASR：**

```bash
docker compose up -d vllm vllm-asr
```

**仅启动 ASR 服务（不启动文本 LLM）：**

```bash
docker compose up -d vllm-asr
```

**查看 ASR 日志：**

```bash
docker logs -f vllm-asr
```

**等待启动完成信号：**

```
INFO:     Application startup complete.
```

冷启动约 5-8 分钟（模型加载 + Triton JIT 编译 + profile_run）。使用 profile 缓存后（Patch 16），后续重启约 90 秒。

**验证服务就绪：**

```bash
curl http://127.0.0.1:8001/v1/models
# 期望返回：{"object": "list", "data": [{"id": "Qwen/Qwen3-ASR-8B", ...}]}
```

### 方式二：docker 手动运行

适合调试、自定义参数或不在 docker compose 管理的场景：

```bash
docker run -d \
  --name vllm-asr \
  --privileged \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --ipc host \
  --shm-size 16gb \
  -p 8001:8000 \
  -v "$VLLM_HOST_MODELS_DIR":/models:ro \
  -v "$PWD/.vllm-cache":/root/.cache/vllm \
  -v "$PWD/.vllm-cache/profile":/root/.cache/vllm-profile \
  -e HF_HOME=/models \
  -e HF_HUB_OFFLINE=1 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_USE_TRITON_AWQ=1 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -e MIOPEN_FIND_MODE=FAST \
  -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  -e VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=25 \
  -e VLLM_SKIP_MEMORY_PROFILING=1 \
  -e VLLM_PROFILE_CACHE_DIR=/root/.cache/vllm-profile \
  vllm-awq4-qwen:builder \
  vllm serve Qwen/Qwen3-ASR-8B \
    --host 0.0.0.0 --port 8000 \
    --supported-tasks transcription,realtime \
    --enforce-eager \
    --max-model-len 8192 \
    --mm-encoder-attn-backend TRITON_ATTN \
    --gpu-memory-utilization 0.9
```

**关键参数说明：**

| 参数 | 说明 |
|---|---|
| `--supported-tasks transcription,realtime` | 注册 `/v1/audio/transcriptions` 和 `/v1/realtime` 端点 |
| `--enforce-eager` | 禁用 HIP 图捕获（gfx1151 上会冻结） |
| `--mm-encoder-attn-backend TRITON_ATTN` | 音频编码器使用 Triton 注意力后端 |
| `--max-model-len 8192` | ASR 不需要 256K 上下文，8192 足够 |
| `--gpu-memory-utilization 0.9` | ASR 模型较小，可高利用率 |

## 调试模式

### 前台运行 + 详细日志

```bash
docker compose up vllm-asr  # 不加 -d，前台运行
```

### 提高日志级别

在 docker-compose.yml 的 ASR 服务中添加：

```yaml
environment:
  VLLM_LOGGING_LEVEL: DEBUG
```

或使用 docker 运行时覆盖：

```bash
docker run ... -e VLLM_LOGGING_LEVEL=DEBUG vllm-awq4-qwen:builder vllm serve ...
```

> **警告**：`DEBUG` 级别会使每个 op 都格式化参数为字符串，导致推理慢 20-100 倍。仅用于调试，不要在生产环境使用。

### 快速测试 — 不启动容器

如果只是想验证模型和音频处理链路（不走 Docker），可以在主机上直接运行：

```bash
# 确保已安装 vLLM 和音频依赖
uv pip install vllm==0.20.0 av soundfile scipy

# 直接启动
vllm serve Qwen/Qwen3-ASR-8B \
  --supported-tasks transcription,realtime \
  --enforce-eager \
  --max-model-len 8192 \
  --mm-encoder-attn-backend TRITON_ATTN
```

> 注意：主机上需要有完整的 ROCm 环境。如果主机已安装 ROCm SDK 且 gfx1151 设备可用，这种方式可以跳过 Docker 构建步骤，直接测试模型。

### 查看 GPU 使用情况

```bash
# GTT 和 VRAM 使用量（Strix Halo UMA 指标）
cat /sys/class/drm/card1/device/mem_info_gtt_used   # 已用 GTT
cat /sys/class/drm/card1/device/mem_info_vram_used  # 已用 VRAM

# 进程 GPU 占用
rocm-smi
```

### 进入运行中容器调试

```bash
docker exec -it vllm-asr bash

# 在容器内：
# 查看已安装依赖
pip list | grep -E 'av|soundfile|scipy|torchaudio'

# 查看 vLLM 环境
python -c "import vllm; print(vllm.__version__)"

# 查看音频模块是否可用
python -c "import av; print('PyAV OK:', av.library_versions)"
python -c "import soundfile; print('soundfile OK')"
python -c "import scipy; print('scipy OK:', scipy.__version__)"

# 手动测试模型加载
python -c "
from vllm import LLM
llm = LLM(
    model='Qwen/Qwen3-ASR-8B',
    enforce_eager=True,
    max_model_len=8192,
    supported_tasks='transcription,realtime',
)
print('Model loaded successfully')
"
```

## 稳定运行模式

### 推荐配置

对于日常使用，建议：

**docker-compose.yml 中 ASR 服务的关键设置：**

| 设置 | 值 | 原因 |
|---|---|---|
| `restart` | `"no"` | 手动控制重启；Strix Halo 是日常驾驶 PC，崩溃后需人工检查 |
| `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` | `25` | 单音频片段上限 25 MB（约 30 分钟 16kHz 单声道 PCM16） |
| `VLLM_SKIP_MEMORY_PROFILING` | `1` | 启用 profile 缓存，后续启动从 ~9 min 降到 ~90 s |
| `--gpu-memory-utilization` | `0.9` | ASR 模型约 16 GB BF16，90% 利用率足够 |
| `--max-model-len` | `8192` | 30s 音频只需要 ~2-3K tokens，8192 有充足余量 |

### 两个服务共存

当 `vllm`（文本 LLM）和 `vllm-asr` 同时运行时：

| 资源 | vllm (文本) | vllm-asr | 总计 |
|---|---|---|---|
| 模型权重 | ~28 GiB | ~16 GiB | ~44 GiB |
| KV cache | ~24 GiB | ~2 GiB | ~26 GiB |
| GTT 总计 | ~50 GiB | ~20 GiB | ~70 GiB |

128 GB UMA 池完全够用。如果还需要运行 RAG api 和 TTS 服务（CPU-only），建议将文本 LLM 的 `gpu_memory_utilization` 降到 0.5，详见 README 中的 multi-stream profile。

### 健康检查

```bash
# 基本健康检查
curl -s http://127.0.0.1:8001/health
# 期望: 200 OK

# 检查模型是否已加载
curl -s http://127.0.0.1:8001/v1/models | python3 -m json.tool

# 快速转录测试
echo "This is a test." | ffmpeg -i - -f s16le -acodec pcm_s16 -ar 16000 -ac 1 -y /tmp/test.wav
# 或者准备一个已有的 .wav 文件
curl -s http://127.0.0.1:8001/v1/audio/transcriptions \
  -F "file=@/tmp/test.wav" -F "model=Qwen/Qwen3-ASR-8B"
```

### 停止和重启

```bash
# 仅停止 ASR（不影响文本 LLM）
docker compose stop vllm-asr

# 重启 ASR
docker compose start vllm-asr

# 完整重启（重建镜像后）
docker compose down
docker compose build
docker compose up -d vllm vllm-asr

# 查看 ASR 日志中的关键信息
docker logs vllm-asr 2>&1 | grep -E "Application startup|Available KV|Cached KV|ERROR"
```

## API 使用指南

### 模式一：非流式转录

```bash
curl http://127.0.0.1:8001/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Qwen/Qwen3-ASR-8B"
```

**响应格式（默认 JSON）：**

```json
{
  "text": "你好世界，这是一段测试音频。"
}
```

**纯文本输出：**

```bash
curl http://127.0.0.1:8001/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Qwen/Qwen3-ASR-8B" \
  -F "response_format=text"
# → 你好世界，这是一段测试音频。
```

**详细输出（verbose_json，含时间戳）：**

```bash
curl http://127.0.0.1:8001/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Qwen/Qwen3-ASR-8B" \
  -F "response_format=verbose_json"
```

**支持指定语言（提高准确率）：**

```bash
curl http://127.0.0.1:8001/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Qwen/Qwen3-ASR-8B" \
  -F "language=zh"
```

**支持的语言（ISO-639-1 代码）：**

`en`（英语）、`zh`（中文）、`ko`（韩语）、`ja`（日语）、`de`（德语）、`ru`（俄语）、`it`（意大利语）、`fr`（法语）、`es`（西班牙语）、`pt`（葡萄牙语）、`ms`（马来语）、`nl`（荷兰语）、`id`（印尼语）、`tr`（土耳其语）、`vi`（越南语）、`yue`（粤语）、`ar`（阿拉伯语）、`ur`（乌尔都语）等。

**支持的音频格式：**

flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm

### 模式二：SSE 流式转录

适合处理较长的音频文件，可以边处理边看到文字输出：

```bash
curl http://127.0.0.1:8001/v1/audio/transcriptions \
  -F "file=@long_audio.wav" \
  -F "model=Qwen/Qwen3-ASR-8B" \
  -F "stream=true"
```

**输出示例（SSE 事件）：**

```
data: {"id": "trsc-xxx", "object": "transcription.chunk", "created": 1234567890, "model": "Qwen/Qwen3-ASR-8B", "choices": [{"delta": {"text": "你好"}, "finish_reason": null}], "usage": null}

data: {"id": "trsc-xxx", "object": "transcription.chunk", "created": 1234567890, "model": "Qwen/Qwen3-ASR-8B", "choices": [{"delta": {"text": "世界"}, "finish_reason": null}], "usage": null}

data: {"id": "trsc-xxx", "object": "transcription.chunk", "created": 1234567890, "model": "Qwen/Qwen3-ASR-8B", "choices": [{"delta": {"text": ""}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": 8, "total_tokens": 8}}
```

**Python 客户端示例：**

```python
import json
import httpx

with httpx.Client(timeout=300.0) as client:
    with open("long_audio.wav", "rb") as f:
        resp = client.post(
            "http://127.0.0.1:8001/v1/audio/transcriptions",
            files={"file": f},
            data={"model": "Qwen/Qwen3-ASR-8B", "stream": "true", "language": "zh"},
        )
        for line in resp.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                delta = event["choices"][0]["delta"]["text"]
                print(delta, end="", flush=True)
                if event["choices"][0]["finish_reason"]:
                    print()
```

### 模式三：WebSocket 实时转录

边说话边出文字，适合麦克风实时输入或音频流场景。

**音频格式要求：**

- 编码：PCM16（16 位整数）
- 采样率：16 kHz
- 声道：单声道（mono）
- 传输方式：base64 编码的原始字节

**Python 客户端示例：**

```python
import asyncio
import json
import base64
import numpy as np
import soundfile as sf
import websockets


async def realtime_transcription(audio_file: str):
    async with websockets.connect("ws://127.0.0.1:8001/v1/realtime") as ws:
        # 1. 接收 session.created
        session = json.loads(await ws.recv())
        print("Session:", session["id"])

        # 2. 选择模型
        await ws.send(json.dumps({
            "type": "session.update",
            "model": "Qwen/Qwen3-ASR-8B",
        }))

        # 3. 读取音频文件为 PCM16
        audio_data, sample_rate = sf.read(audio_file, dtype="int16")
        # 如果是立体声，转单声道
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1).astype(np.int16)
        # 如果采样率不是 16kHz，需要重采样
        if sample_rate != 16000:
            from scipy.signal import resample_poly
            audio_data = resample_poly(audio_data, 16000, sample_rate).astype(np.int16)

        # 4. 发送音频块（每块约 0.1 秒 = 1600 samples）
        chunk_size = 1600
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            audio_bytes = chunk.tobytes()
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_bytes).decode("ascii"),
            }))
            await asyncio.sleep(0.01)  # 模拟实时流式发送

        # 5. 触发识别
        await ws.send(json.dumps({
            "type": "input_audio_buffer.commit",
        }))

        # 6. 接收识别结果
        full_text = ""
        async for msg in ws:
            event = json.loads(msg)
            if event["type"] == "transcription.delta":
                delta = event["delta"]["text"]
                full_text += delta
                print(delta, end="", flush=True)
            elif event["type"] == "transcription.done":
                print()
                print(f"完整文本: {full_text}")
                print(f"Token 使用: {event['usage']}")
                break


if __name__ == "__main__":
    asyncio.run(realtime_transcription("test.wav"))
```

**实时麦克风输入（使用 sounddevice）：**

```python
import asyncio
import base64
import json

import numpy as np
import sounddevice as sd
import websockets


async def mic_transcription():
    # 音频缓冲区
    audio_queue = asyncio.Queue()

    def audio_callback(indata, frames, time_info, status):
        """sounddevice 回调，将 PCM16 数据放入队列"""
        # 转 int16（PCM16）
        pcm16 = (indata * 32767).astype(np.int16).tobytes()
        audio_queue.put_nowait(pcm16)

    # 启动麦克风采集
    stream = sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=16000,
        dtype="float32",
        blocksize=1600,  # ~100ms per block
    )
    stream.start()

    async with websockets.connect("ws://127.0.0.1:8001/v1/realtime") as ws:
        await ws.recv()  # session.created

        await ws.send(json.dumps({
            "type": "session.update",
            "model": "Qwen/Qwen3-ASR-8B",
        }))

        # 持续发送麦克风数据
        async def send_audio():
            while True:
                pcm16 = await audio_queue.get()
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm16).decode("ascii"),
                }))

        # 启动音频发送任务
        sender = asyncio.create_task(send_audio())

        try:
            # 接收转录结果
            full_text = ""
            while True:
                msg = await ws.recv()
                event = json.loads(msg)
                if event["type"] == "transcription.delta":
                    delta = event["delta"]["text"]
                    full_text += delta
                    print(delta, end="", flush=True)
                elif event["type"] == "transcription.done":
                    print(f"\n[本轮识别完成] {full_text}")
                    full_text = ""
                    # 发送 commit 触发下一轮识别
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.commit",
                    }))
        except KeyboardInterrupt:
            print("\n停止录制")
            sender.cancel()
            stream.stop()
            stream.close()


if __name__ == "__main__":
    asyncio.run(mic_transcription())
```

### 音频格式转换

如果你的音频文件不是 16kHz PCM16 单声道，可以用 ffmpeg 转换：

```bash
# 转换为 16kHz 单声道 WAV（PCM16）
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f wav output.wav

# 转换为原始 PCM16（无 WAV 头）
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f s16le -acodec pcm_s16 output.pcm

# 使用 PCM 文件测试 WebSocket
python3 -c "
import base64, asyncio, json, websockets

async def send_pcm():
    async with websockets.connect('ws://127.0.0.1:8001/v1/realtime') as ws:
        await ws.recv()  # session.created
        await ws.send(json.dumps({'type': 'session.update', 'model': 'Qwen/Qwen3-ASR-8B'}))
        with open('output.pcm', 'rb') as f:
            while chunk := f.read(3200):  # 每块 ~100ms
                await ws.send(json.dumps({
                    'type': 'input_audio_buffer.append',
                    'audio': base64.b64encode(chunk).decode()
                }))
                await asyncio.sleep(0.05)
        await ws.send(json.dumps({'type': 'input_audio_buffer.commit'}))
        async for msg in ws:
            event = json.loads(msg)
            if event['type'] in ('transcription.delta', 'transcription.done'):
                print(event)

asyncio.run(send_pcm())
"
```

## 故障排查

### 问题：服务启动后 /v1/audio/transcriptions 返回 404

**原因：** `--supported-tasks` 未包含 `transcription`。

**解决：** 检查启动命令中是否包含 `--supported-tasks transcription,realtime`。

### 问题：音频上传后返回错误 "Invalid or unsupported audio format"

**原因：** 音频格式不被 PyAV 或 soundfile 支持，或文件已损坏。

**解决：**
1. 确认音频格式在支持列表中（flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm）
2. 用 ffmpeg 转换为 WAV 再试：`ffmpeg -i input.ext -ar 16000 -ac 1 output.wav`
3. 检查容器内是否安装了音频依赖：`docker exec vllm-asr python -c "import av, soundfile; print('OK')"`

### 问题：WebSocket 连接后无 transcription.delta 输出

**原因：**
1. 未发送 `session.update` 选择模型
2. 音频格式不正确（非 PCM16、非 16kHz、非 mono）
3. 音频数据量太小（< 1 秒）

**解决：**
1. 确保连接后先发送 `{"type": "session.update", "model": "Qwen/Qwen3-ASR-8B"}`
2. 确认音频是 16kHz PCM16 单声道
3. 发送至少 2-3 秒的音频再发送 `commit`

### 问题：GPU 显存不足 OOM

**原因：** 文本 LLM 和 ASR 同时运行且显存利用率设置过高。

**解决：**
1. 降低文本 LLM 的 `--gpu-memory-utilization` 到 0.5-0.7
2. 或降低 ASR 的 `--gpu-memory-utilization` 到 0.6
3. 查看当前显存使用：`cat /sys/class/drm/card1/device/mem_info_gtt_used`

### 问题：启动慢（> 10 分钟）

**原因：** 首次启动需要 profile_run（Triton JIT 编译 + 合成推理），约 5-8 分钟。

**解决：**
1. 确保 `VLLM_SKIP_MEMORY_PROFILING=1` 已设置
2. 确保 `./.vllm-cache/profile/` 目录存在且可写
3. 第二次启动会读取 profile 缓存，降到 ~90 秒
4. 如果修改了 `max_model_len`、`gpu_memory_utilization` 等配置，会重新 profile（缓存 key 变了）

### 问题：模型下载失败 "Access denied"

**原因：** Qwen3-ASR-8B 是 gated model，需要 HF 账号接受模型条款。

**解决：**
1. 登录 https://huggingface.co/Qwen/Qwen3-ASR-8B，点击 "Agree and access repository"
2. 确保 `.env` 中的 `HF_TOKEN` 有效
3. 重新运行下载命令
