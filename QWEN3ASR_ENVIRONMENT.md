# Qwen3-ASR / vLLM 独立环境

本目录 `qwen3asr_vllm/` 是与 MiniCPM `.venv` 分开的 Qwen3-ASR 运行环境。实际可运行环境位于：

```text
qwen3asr_vllm/isolated_vllm014_torch291
```

它使用 Torch 2.9.1+cu128，并把 Qwen/vLLM 所需包放在独立目录，不会改写 MiniCPM 的 Torch 2.8.0 和 Transformers 4.51.0。

## 已安装版本

- Python 3.10.12
- NVIDIA GPU: A800-SXM4-80GB
- CUDA runtime visible to PyTorch: 12.8
- Torch: 2.9.1+cu128
- torchaudio: 2.9.1+cu128
- torchvision: 0.24.1+cu128
- `qwen-asr==0.0.6`
- `vllm==0.14.0`
- `transformers==4.57.6`
- `tokenizers==0.22.2`
- `nagisa==0.2.11`
- `soynlp==0.0.493`
- `qwen-omni-utils==0.0.9`

这里没有强制安装 CUDA 13。vLLM 0.14.0 与 Torch 2.9.1+cu128 已经完成真实 GPU 加载和推理：

```text
torch 2.9.1+cu128 12.8 True
vllm 0.14.0
checkpoint load: 2 shards
ASR result: 当出现《植物大战僵尸》的时候，提醒我。
```

## 使用

```bash
cd /root/paddlejob/workspace/env_run/avatar_full/minicpm_official_rag_test
qwen3asr_vllm/isolated_vllm014_torch291/bin/python \
  -c 'import torch, vllm, qwen_asr; print(torch.__version__, vllm.__version__)'
```

这个目录采用目标目录式隔离，因为系统 Python 的 `ensurepip`/virtualenv 不完整。入口脚本会依次加载：

```text
isolated_vllm014_torch291/deps
isolated_vllm014_torch291/site-packages
qwen3asr_vllm/site-packages
avatar_full/.venv/site-packages
```

不要用普通 `pip` 修改 MiniCPM `.venv`。

## 与 MiniCPM 服务联调

当前仓库中的 `Qwen3ASRStreaming` 需要一个提供以下方法的 backend：

```text
init_streaming_state(...)
streaming_transcribe(audio_chunk, state)
finish_streaming_transcribe(state)
```

适配器会按约 2 秒音频块调用 ASR，保留 partial，只有 final transcript 才触发 RAG；MiniCPM 仍由单独 owner 线程全程以 AUDIO 模式运行。

真实 HTTP 服务使用 Qwen 包自带的 streaming demo，把 `Qwen3ASRModel.LLM` 与 `/api/start`、`/api/chunk`、`/api/finish` 组合在一起。启动命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_HOST_IP=127.0.0.1 VLLM_PORT=18580 \
NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
qwen3asr_vllm/isolated_vllm014_torch291/bin/python \
  -m qwen_asr.cli.demo_streaming \
  --asr-model-path "$(pwd)/models/Qwen3-ASR-1.7B" \
  --host 127.0.0.1 --port 18180 \
  --gpu-memory-utilization 0.50 \
  --chunk-size-sec 2.0 --unfixed-chunk-num 2 --unfixed-token-num 5
```

服务启动后验证：

```bash
qwen3asr_vllm/isolated_vllm014_torch291/bin/python \
  qwen3asr_vllm/test_http_service.py
```

## 2026-11-17 实机测试结果

Qwen3-ASR-1.7B 权重位于 `models/Qwen3-ASR-1.7B`，约 4.4 GB。

vLLM 版本结论：

- vLLM 0.11.0 能导入，但没有 Qwen3-ASR 自定义模型架构。
- vLLM 0.13.0 配合 Torch 2.9.0+cu128 能加载 `_C`，但缺少 Qwen backend 使用的 0.14 模块布局。
- vLLM 0.14.0 配合 Torch 2.8 会在加载 `_C` 时触发 `std::bad_alloc`，属于二进制 ABI 不匹配。
- vLLM 0.14.0 配合 Torch 2.9.1+cu128 可以加载权重并执行真实推理。13.7 秒示例音频的推理时间约 0.93 秒。

绕过 vLLM、直接用 Qwen 包内 Transformers backend 加载权重也可以运行：

```bash
qwen3asr_vllm/bin/python qwen3asr_vllm/test_transformers_asr.py
```

该入口会延迟导入 `qwen_asr` 顶层包，避免其可选 vLLM backend 先被导入。实测同样识别为
“当出现《植物大战僵尸》的时候，提醒我。”

流式 HTTP 服务实测：

```text
chunk 1: 当出现植物大战。
chunk 2: 当出现《植物大战僵尸》的时候，提醒我。
final:   当出现《植物大战僵尸》的时候，提醒我。
```

`Qwen3ASRHTTPBackend` 已接入 `Qwen3ASRStreaming` 和 `ASRRetrievalWorker`。实测在 MiniCPM RAG 注入前完成了：

```text
真实 Qwen3-ASR HTTP final transcript
→ RAG 查询
→ 命中 plant-zombies
→ 生成 AUDIO prefill 的 text_list
→ 保持 force_listen 注入
```

测试 runtime 的注入事件链为 `asr.final → retrieval.result → rag.injected`，注入文本包含 `<reference>`，末尾保留单个换行。

完整 MiniCPM 实模型测试目前停在独立的资源检查：当前 `model/assets/token2wav` 缺 `config.yaml` 和 `hift_original.pt`，因此尚未执行真实 MiniCPM 出声。该问题不影响 Qwen3-ASR 服务、ASR→RAG 调用和语音模式注入状态机的验证。
