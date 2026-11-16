# 官方 MiniCPM-o 全双工 text_list 独立验证

已新增基于官方 `text_list` 的本地 RAG 检索和生成入口 `run_rag.py`。
使用方法见 [RAG_GUIDE.md](RAG_GUIDE.md)，本次结果见 [RAG_REPORT.md](RAG_REPORT.md)。
以下内容保留原入口对照实验；其中 GPU 2 是此前实验记录。新增 RAG 验证使用 GPU 0。

这是模型入口实验，不启动数字人、网关、浏览器或生产服务。源码来自
`../minicpm_demo/model`，与此前核查的官方 revision
`fb09628e98b1f55d8d5b863e0eccfcdd11bff499` 的 `modeling_minicpmo.py` 校验一致。
本目录没有修改模型源码，没有使用原实验中的私有 knowledge injection 或 force-speak。

完整结果见 [REPORT.md](REPORT.md)。

## 资源与隔离

- 小型代码、配置、tokenizer 文件复制到 `model/`。
- 四个大权重分片及 `assets/` 链接到现有 `minicpm_demo/model/`，只读加载。
- 参考音色复制自 `minicpm_demo/assets/ref_audio/ref_minicpm_signature.wav`。
- `assets/knowledge_original.json` 是原知识文件快照；600 元来自其中的差旅条目。
  437 元是为了验证资料覆盖能力而构造的测试金额，不是政策更新。
- 真人语音来自 demo 已有的“当出现植物大战僵尸的时候提醒我.wav”。
- ASR 复用 `../minicpm_rag_test/models/whisper-small`，不提供答案提示。
- 全程离线加载，不下载权重或新增依赖；复用根目录 `.venv`。
- 本次使用物理 GPU 2。开始时 GPU 0 已占用约 66 GB；没有终止其他进程。

不要编辑链接指向的大文件。需要换权重或 Token2Wav 实现时，应另建副本。
来源、文件大小及小文件 SHA256 见 `source_manifest.json`。

## 运行

先用 `nvidia-smi` 确认目标 GPU 空闲。以下是本次所用命令；如需改用 GPU 0，
将 `CUDA_VISIBLE_DEVICES=2` 改为 `0`。

```bash
cd /root/paddlejob/workspace/env_run/avatar_full/minicpm_official_rag_test

# 10 个场景，11 份回答：原版 text_list、音频混合、跨轮更新和真人语音。
CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python -u run_probe.py

# 从实际生成的 WAV 独立转写；不将生成文字/预期金额喂给 ASR。
CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python -u transcribe_outputs.py

# 3 个补充场景：冷启动混合输入及无依据拒答。
CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python -u run_probe.py \
  --cases cold_mixed_437,cold_mixed_refreshed_437,ongoing_no_evidence \
  --output outputs/controls

CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python -u transcribe_outputs.py \
  --results outputs/controls/results.json

# 评价规则单测。
../.venv/bin/python -m pytest test_evaluation.py -q
```

脚本自动设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。
默认输出目录为 `outputs/isolated/`；重复执行会覆盖同名结果，保留一次实验时
可用 `--output outputs/<新名称>` 指定新目录。

只跑最有用的正常会话注入用例：

```bash
CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python -u run_probe.py \
  --cases ongoing_437,audio_then_text \
  --output outputs/boundary_injection

CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python -u transcribe_outputs.py \
  --results outputs/boundary_injection/results.json
```

## 实验设计

- 直接调用 `AutoModel.from_pretrained(...).as_duplex()`，然后使用官方
  `prepare / streaming_prefill / streaming_generate`。
- 使用 BF16、SDPA、greedy 文本解码、seed 42，单次最多 20 个生成步骤，
  每份回答最多 16 个 unit；后续 unit 正常送入 1 秒静音，不重复注入证据。
- 正常会话用例先用公开的 `force_listen_count=1` 建立一个监听 unit。
  真人语音用例也用此公开配置保持提问输入阶段的监听。没有强制说话。
- 每个独立场景先调用底层 `model.reset_session()`，再 `duplex.prepare()`，
  清除上一个场景的音频 encoder KV；同一场景的多轮更新保留会话上下文。
  **实际同一会话内注入下一轮资料时，不应每次 reset 或 prepare。**
- 观测包装器记录 feed 的长度、KV 增长、是否返回 logits，并对最后一次
  forward 的完整 hidden states 使用同一个 lm_head 计算比较值。
  普通用例不改变官方 feed 的参数或返回值。
- 名称带 `refreshed` 的用例是明确标记的诊断对照：将
  `pending_logits` 替换成最后一次文本 feed 对应的 logits。
  这是运行时干预，不是官方原版行为；模型文件仍保持不变。

## 输出

每组输出包含：

- `results.json`：输入、逐 unit 文本和控制 token、KV/feed 轨迹、耗时、
  音频统计、是否自然结束、金额或关键词检查。
- `*.wav`：实际输出，24 kHz float32；排除 listen 返回的占位静音。
- `asr_results.json`：Whisper-small 回转文字与金额检查。

`outputs/official/` 是最初探索，未完全隔离音频 encoder KV，且 warmup
自然生成了发言；仅保留排查记录。结论使用 `outputs/isolated/` 和
`outputs/controls/`，不要将探索组的组间差异当作对照证据。

脚本退出 0 只表示模型调用完成，不表示 RAG 语义或语音质量全部通过。
金额匹配仅是明确的字符串规则；ASR 有同音字和数字识别误差，不代替人工听验。
观测 hook、额外 lm_head 和 CUDA 同步会影响耗时，本实验不是实时性能基准。
