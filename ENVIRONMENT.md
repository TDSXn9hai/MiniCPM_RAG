# 本地虚拟环境

本目录的 `.venv` 是 MiniCPM RAG 运行环境，使用 `virtualenv --system-site-packages` 创建，
并通过 `avatar_full_existing.pth` 复用工作区已经验证的 CUDA 栈：

- Python 3.10.12
- PyTorch 2.8.0+cu128
- Transformers 4.51.0
- librosa 0.11.0
- MiniCPM-o 官方本地模型依赖

安装基础依赖：

```bash
/usr/local/bin/python -m pip --python .venv/bin/python install \
  --no-deps -r requirements-retrieval.txt \
  transformers==4.51.0 accelerate==1.12.0 safetensors==0.7.0 \
  librosa==0.11.0 soundfile==0.12.1 einops==0.8.1 httpx==0.28.1
```

`minicpmo-utils` 已单独以 `--no-deps` 安装。它的发布元数据固定要求
`librosa==0.9.0`，与当前官方 MiniCPM 实际使用的 `librosa>=0.11` 冲突，因此不要让
pip 重新解析它的依赖。

## Qwen3-ASR

Qwen3-ASR 流式接口使用独立环境，不能与当前 MiniCPM `.venv` 混装：

- `qwen-asr==0.0.6` 要求 `transformers==4.57.6`，会覆盖 MiniCPM 的 4.51.0。
- `vllm==0.14.0` 要求 `torch==2.9.1`，而 MiniCPM 验证环境使用 Torch 2.8.0+cu128。

`requirements-qwen3-asr.txt` 是独立环境清单，不应安装到 `.venv`。真实验证使用
Torch 2.9.1+cu128、vLLM 0.14.0 和 Transformers 4.57.6，并已完成
Qwen3-ASR-1.7B 权重加载、流式 HTTP 服务及 final transcript RAG 检索。

完整版本矩阵、启动命令、已验证输出和当前运行边界见
[QWEN3ASR_ENVIRONMENT.md](QWEN3ASR_ENVIRONMENT.md)。模型权重和整个 Qwen
虚拟环境不随仓库上传。

## 流式语音 RAG 控制台

`.venv` 额外安装 `requirements-console.txt`：

```bash
uv pip install --python .venv/bin/python -r requirements-console.txt
```

控制台使用 FastAPI/uvicorn WebSocket、AudioWorklet 和 Web Audio API。
输入 PCM 固定为 16 kHz mono float32，每块 16000 samples（1 秒）；
输出为 24 kHz mono float32 PCM。输入和输出音频都不写盘。

Qwen ASR HTTP 服务继续使用独立 `.venv-qwen`，控制台主进程使用 `.venv`。
两个进程建议放在不同 GPU 上，并通过 `--qwen-asr-url` 连接。
