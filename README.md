# MiniCPM RAG：官方全双工 text_list 入口

使用 MiniCPM-o 4.5 官方全双工接口，**后台 ASR/RAG 与 MiniCPM 同卡并行，应用全程保持 AUDIO**。
检索证据与问题通过 `streaming_prefill(audio_waveform=..., text_list=[...])` 合并输入，
注入所在 unit 保持强制监听，下一 AUDIO unit 才释放监听并生成文字与语音。
没有独立 TEXT prefill、私有 KV cache 注入或官方模型源码修改。

本仓库包含完整应用代码、知识样例、单元测试、集成测试和实验文档。
模型权重、模型目录、运行输出、虚拟环境与缓存均不上传。

## 安装

验证环境为 Linux、Python 3.10、CUDA GPU，PyTorch 2.8.0、Transformers 4.51.0。
原测试使用 A800 80GB；不代表其他硬件已经验证。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
# 请根据自己的驱动/CUDA 环境安装适配的 PyTorch；原测试版本为 2.8.0。
pip install -r requirements.txt
```

纯检索和单元测试可只安装轻量依赖：

```bash
pip install -r requirements-retrieval.txt
python run_rag.py --query '上海出差住宿能报多少' --retrieve-only
python -m pytest tests test_evaluation.py -q
```

## 准备模型

请将完整官方 MiniCPM-o 4.5 模型目录放到 `model/`，或通过 `--model-path` 指定。
目录必须包含官方 Python 实现、配置、tokenizer、全部权重分片及 Token2Wav 所需 assets，
不能只拷贝 safetensors。原验证采用官方 revision：
`fb09628e98b1f55d8d5b863e0eccfcdd11bff499`。
来源清单见 `source_manifest.json`，其中绝对路径仅记录原实验环境，不要求本机存在。

可复用已有完整目录；没有模型时，按官方模型说明获取：
https://huggingface.co/openbmb/MiniCPM-o-4_5

本项目运行时强制离线加载，不会自动下载缺失模型。`trust_remote_code=True` 会执行模型目录中的
官方 Python 实现，应使用可信官方文件。参考音色 `assets/ref_audio.wav` 来自原 MiniCPM demo，
模型及其资源的使用条件以各自上游授权为准，本仓库不重新授予模型授权。

## 运行

音频入口需要完整本地 Whisper-small，默认放在 `models/whisper-small/`，也可通过
`--asr-model` 指定。Whisper 和 MiniCPM 默认均常驻逻辑 `cuda:0`；启动阶段顺序加载，
每轮推理通过后台线程和独立 CUDA stream 并行。TF-IDF 检索在后台 CPU 执行。

```bash
# 同卡异步 ASR/RAG + AUDIO 监听和注入。
CUDA_VISIBLE_DEVICES=0 python -u run_rag.py \
  --audio /path/to/question.wav \
  --model-path /path/to/MiniCPM-o-4_5 \
  --asr-model /path/to/whisper-small --device cuda:0

# 仅检索，不需要模型或 GPU。
python run_rag.py --query '上海出差住宿能报多少' --retrieve-only

# 真实模型文字/语音生成。请先确认目标 GPU 空闲。
CUDA_VISIBLE_DEVICES=0 python -u run_rag.py \
  --model-path /path/to/MiniCPM-o-4_5 \
  --knowledge knowledge/sample.json \
  --query '上海出差住宿能报多少'

# 同一模型会话连续提问，知识文件每轮检查更新。
CUDA_VISIBLE_DEVICES=0 python -u run_rag.py \
  --model-path /path/to/MiniCPM-o-4_5 --interactive
```

交互时 `/reset` 重置模型会话，`/quit` 退出。可重复使用 `--query` 顺序提问。
默认模型路径为 `model/`，默认知识文件为 `knowledge/sample.json`。
使用原工程知识文件时传 `--knowledge /path/to/minicpm_rag_test/knowledge/sample.json`。
默认输出 `outputs/rag/<UTC时间>/`；用 `--output` 指定新目录，已有结果时拒绝覆盖。

## 数据流和代码

```text
后台 worker：完整音频 → Whisper ASR → TF-IDF RAG → Future
MiniCPM owner：音频 AUDIO + force_listen → 等待时静音 AUDIO + force_listen
→ AUDIO + text_list 注入，继续 listen → 下一 AUDIO unit 释放 → 分块文字/24kHz WAV
```

| 文件 | 用途 |
|---|---|
| `rag_core.py` | 从原 minicpm_rag_test 迁移的检索、证据卡与引用结构 |
| `rag_pipeline.py` | 文件校验、SHA256 热更新、弱命中过滤、无证据拒答及生成编排 |
| `async_audio_rag.py` | 常驻 Whisper、后台 ASR/RAG worker 和并行流水线 |
| `official_runtime.py` | 全 AUDIO API 适配、监听门控、会话管理和音频流输出 |
| `run_rag.py` | 可直接运行的 RAG CLI |
| `tests/` | 检索/API/CLI 单元测试与真实 GPU 集成驱动 |
| `run_probe.py` | 原官方入口与混合输入诊断实验，含明确标注的对照干预 |
| `transcribe_outputs.py` | 生成 WAV 的独立 Whisper 转写 |

新增 RAG 运行路径不调用诊断实验中的 pending_logits 干预。
原 run_probe.py 的音频诊断用例需通过 `--user-audio /path/to/fixture.wav` 提供输入音频。
检索参数、接口、会话约束及输出格式见 [RAG_GUIDE.md](RAG_GUIDE.md)。

## 验证结果和边界

**31 项自动化测试通过**。原工作区中已完成真实同 GPU 并行、检索等待门控和多轮热更新验证：

- 自然返回时序：ASR 约 0.58 秒与 MiniCPM AUDIO 处理区间重叠，检索也有重叠；模型正常回答，
  但只是泛化应答，没有展开资料细节。
- 受控延迟检索：等待两个 AUDIO unit、注入 unit 仍保持监听，之后回答包含全部目标资料细节。
- 文字入口改为 AUDIO 后，同一会话 600→437 元知识热更新和无证据拒答通过。
- 生成 WAV 的独立转写识别到资料细节和金额，存在同音字误识别。

两个模型同时 `from_pretrained` 曾出现 meta tensor 错误，现已改为顺序启动构造，
随后推理并行。该回归已加入测试。历史独立 TEXT 和音频空回答结果保留在旧报告中。

**模型使用证据的质量仍有格式/时序敏感性**。当前是完整音频文件后台 ASR 与 MiniCPM
音频分块处理重叠，不是麦克风增量转写服务；尚未实现 VAD、WebSocket、用户打断或严格实时帧调度。
独立 CUDA stream 允许同卡调度，不保证 kernel 必然同时运行。词面检索分数不是可信概率，
引用标识送入的证据，不表示已验证每句回答。

文件最多30秒。`--asr-device` 如显式设置必须与 `--device` 一致；不再以 CPU ASR 运行并行音频路径。
`--retrieval-timeout 120` 设置等待超时，`--wait-unit-seconds 1` 控制静音等待 unit 间隔。
音频无证据时保持监听并返回固定文字拒答，不生成拒答 WAV；此时 model_called=true、generation_called=false。
文字无证据时不调用 MiniCPM。

更多资料：

- [异步同卡实现验证](ASYNC_AUDIO_RAG_REPORT.md)：最新实现、重叠时间线及真实结果。
- [AUDIO 门控与换行对照](AUDIO_GATE_NEWLINE_REPORT.md)、[末段混合输入对照](AUDIO_TAIL_RAG_REPORT.md)。
- [旧版 RAG 验证记录](RAG_REPORT.md)：串行/独立 TEXT 入口的历史结果。
- [原官方入口实验说明](PROBE_README.md)及[详细报告](REPORT.md)：包含混合输入问题的对照。
- `outputs/` 下的路径是历史运行记录位置，原始输出不随仓库上传。

本仓库是从 avatar_full/minicpm_official_rag_test 整理的独立版本。
历史文档中的原工作区路径/GPU 编号仅作实验溯源；新环境请优先按本 README 操作。

## 运行验证

```bash
python -m pytest tests test_evaluation.py -q
# 需已准备 model/ 和空闲 GPU，固定写入 outputs/rag_integration/；已有目录会拒绝重跑。
CUDA_VISIBLE_DEVICES=0 python -u tests/integration_rag.py --output outputs/new_text_regression
# 真实同卡并行及人为延迟检索门控，需完整 model/，输入使用自己的 WAV。
CUDA_VISIBLE_DEVICES=0 python -u tests/integration_async_audio.py \
  --audio /path/to/question.wav --asr-model /path/to/whisper-small
# 从真实输出 WAV 独立转写，需要本地 Whisper-small。
CUDA_VISIBLE_DEVICES=0 python transcribe_outputs.py \
  --results outputs/rag_integration/results.json --model /path/to/whisper-small
```

本次发布前对整理后的仓库运行了单元测试；没有在独立发布目录重复加载权重执行 GPU 实验。

末段混合输入和换行诊断脚本可显式运行：

```bash
python test_audio_tail_rag.py --audio /path/to/question.wav --asr-model /path/to/whisper-small
python probe_audio_gate_newline.py --audio /path/to/question.wav --asr-model /path/to/whisper-small
```

两个诊断脚本固定保留各自 outputs 子目录，已有目录时拒绝覆盖；与原 run_probe.py 一样，
包含实验对照的逻辑，不能替代正式 run_rag.py。pytest.ini 使默认测试只收集单元测试。
