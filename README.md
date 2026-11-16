# MiniCPM RAG：官方全双工 text_list 入口

使用 MiniCPM-o 4.5 官方 `streaming_prefill(text_list=[...])` 输入检索证据与用户问题，
通过 `streaming_generate()` 输出文字与语音。没有私有 KV cache 注入，也不修改官方模型。

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

```bash
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
问题文本 → RAGRetriever → TF-IDF 检索/过滤 → 证据卡
→ 独立官方 text_list unit → streaming_generate → 分块文字和 24kHz WAV
```

| 文件 | 用途 |
|---|---|
| `rag_core.py` | 从原 minicpm_rag_test 迁移的检索、证据卡与引用结构 |
| `rag_pipeline.py` | 文件校验、SHA256 热更新、弱命中过滤、无证据拒答及生成编排 |
| `official_runtime.py` | 官方全双工 API 适配、会话管理和音频流输出 |
| `run_rag.py` | 可直接运行的 RAG CLI |
| `tests/` | 检索/API/CLI 单元测试与真实 GPU 集成驱动 |
| `run_probe.py` | 原官方入口与混合输入诊断实验，含明确标注的对照干预 |
| `transcribe_outputs.py` | 生成 WAV 的独立 Whisper 转写 |

新增 RAG 运行路径不调用诊断实验中的 pending_logits 干预。
原 run_probe.py 的音频诊断用例需通过 `--user-audio /path/to/fixture.wav` 提供输入音频。
检索参数、接口、会话约束及输出格式见 [RAG_GUIDE.md](RAG_GUIDE.md)。

## 验证结果和边界

22 项自动化测试通过。真实模型在同一会话中检索原知识回答 600 元，修改测试知识副本后
回答 437 元；无关问题由应用层直接拒答，不调用模型。生成 WAV 的独立转写金额一致，
但存在同音字识别误差。金额修改只是实验数据，不代表真实政策。

**音频文件端到端生成尚未通过**：真实音频 ASR 与检索正确，但随后模型空回答结束，
已记录为 incomplete。当前通过验证的是文本问题驱动的官方全双工生成。
CLI 串行运行，尚未实现实时麦克风、用户打断和检索与收音并行调度。
检索采用词面相似度，分数不是可信概率；引用标识送入的证据，不表示已验证每句回答。

音频实验需要另备本地 Whisper-small，默认放在 `models/whisper-small/`：

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_rag.py --audio /path/to/question.wav \
  --asr-model /path/to/whisper-small --asr-device cpu \
  --model-path /path/to/MiniCPM-o-4_5
```

音频最多 30 秒。默认 no_evidence 为固定文字拒答，不生成 WAV。

更多资料：

- [RAG 实现验证记录](RAG_REPORT.md)：成功结果、失败结果和测试依据。
- [原官方入口实验说明](PROBE_README.md)及[详细报告](REPORT.md)：包含混合输入问题的对照。
- `outputs/` 下的路径是历史运行记录位置，原始输出不随仓库上传。

本仓库是从 avatar_full/minicpm_official_rag_test 整理的独立版本。
历史文档中的原工作区路径/GPU 编号仅作实验溯源；新环境请优先按本 README 操作。

## 运行验证

```bash
python -m pytest tests test_evaluation.py -q
# 需已准备 model/ 和空闲 GPU，固定写入 outputs/rag_integration/；已有目录会拒绝重跑。
CUDA_VISIBLE_DEVICES=0 python -u tests/integration_rag.py
# 从真实输出 WAV 独立转写，需要本地 Whisper-small。
CUDA_VISIBLE_DEVICES=0 python transcribe_outputs.py \
  --results outputs/rag_integration/results.json --model /path/to/whisper-small
```

本次发布前对整理后的仓库运行了单元测试；没有在独立发布目录重复加载权重执行 GPU 实验。
