# 官方全双工 AUDIO 模式的并行 ASR / RAG

`run_rag.py --audio` 现在将 ASR 和检索放在后台 worker，与 MiniCPM AUDIO 处理并行。
Whisper-small 和 MiniCPM 默认常驻同一逻辑 GPU `cuda:0`。模型源码未修改，不使用私有 KV 注入。
文字问题、音频问题及 RAG 注入在应用运行路径中均使用 AUDIO，没有独立 TEXT prefill。
原 `run_probe.py` 等诊断脚本保留历史 TEXT 对照，不能把它们当作当前应用入口。

## 运行

在本独立仓库根目录执行，先按 README 安装依赖并准备本地模型，确认目标 GPU 空闲：

```bash
# 同卡并行音频入口；ASR 不再先串行执行。
CUDA_VISIBLE_DEVICES=0 python -u run_rag.py \
  --audio /path/to/question.wav \
  --knowledge knowledge/sample.json \
  --device cuda:0

# 文字问题同样使用静音 AUDIO + text_list 注入。
CUDA_VISIBLE_DEVICES=0 python -u run_rag.py \
  --knowledge knowledge/sample.json \
  --query '上海出差住宿能报多少'

# 多轮文字交互，/reset 重建会话，/quit 退出。
CUDA_VISIBLE_DEVICES=0 python -u run_rag.py --interactive

# 仅检索，无 MiniCPM 模型调用。
python run_rag.py \
  --query '上海出差住宿能报多少' --retrieve-only
```

`CUDA_VISIBLE_DEVICES=0` 选择物理 GPU 0，`--device cuda:0` 是进程内的逻辑编号。
音频并行入口如指定 `--asr-device`，必须与 `--device` 一致；不再用 `--asr-device cpu`。
`--asr-model` 可指定现有 Whisper-small；默认读取本仓库 models/whisper-small/。
`--model-path` 默认本项目 model/，`--ref-audio` 默认 assets/ref_audio.wav。
模型必须提前准备，运行时离线加载，不自动下载。

默认知识文件为本项目 knowledge/sample.json 独立副本；编辑原工程知识时必须指定 `--knowledge`。
默认输出 outputs/rag/<UTC时间>/；可用 `--output` 指定新目录，已有 results.json 时拒绝覆盖。

## 并行与监听门控

```text
启动：顺序加载 Whisper 与 MiniCPM，两个模型常驻同一 GPU
每轮：
  后台 worker：完整音频 → Whisper GPU ASR → CPU TF-IDF RAG → Future
  MiniCPM owner：音频分块 AUDIO / force_listen
                → 若结果未到，静音 AUDIO / force_listen 等待
                → Future 完成后，静音 AUDIO + text_list 注入 / force_listen
                → 下一静音 AUDIO unit 释放监听 → 正常生成文字/语音
```

Whisper 有独立线程及 CUDA stream；MiniCPM 的模型、会话、prefill/generate 只由一个 owner 操作。
检索依赖转写结果，所以 ASR→RAG 在 worker 内顺序执行；两者都独立于 MiniCPM owner。
启动时不并发调用两个模型的 from_pretrained：本次实测并发构造出现 meta tensor 错误，
已改为顺序构造。启动完成后每轮推理并行，Whisper 不会每轮卸载或重新加载。
CPU 检索没有独立占用另一张 GPU。

官方混合输入分支先处理音频、再处理文本，但首步 pending_logits 仍来自音频。
应用在注入 unit 强制 listen，直到文本已经写入，再由下一 AUDIO forward 正常生成。
没有更改 pending_logits，也没有强制 speak。prompt 末尾固定一个换行，依据此前对照选择；
这一格式并不保证所有问法都使用证据，实际回答质量仍需评估。

参数 `--retrieval-timeout 120` 限制等待，`--wait-unit-seconds 1` 控制静音等待 unit 间隔。
等待使用 Future 的超时等待，释放 GIL，不高速循环塞入大量静音。
输入文件最多30秒；后台 ASR 使用完整文件，并非边收音边产生 partial transcript。
文件读完后，MiniCPM 对音频分块处理与后台任务重叠；没有新增麦克风、VAD、WebSocket 或用户打断服务。
不宣称达到严格实时帧调度或 GPU kernel 必然同时运行。

## 失败处理和会话

新会话调用底层 reset_session，再 prepare，prime 一个 AUDIO 监听 unit。
正常多轮保留模型会话；异常、未完成生成或提前关闭流会清理 active 状态，并按需要重建会话。
同一 runtime / audio pipeline 同时只接受一个 turn，后台不会直接修改 MiniCPM。

无证据：应用固定文字拒答，不释放 MiniCPM 说话，不生成拒答 WAV。
音频路径此时 model_called=true（已经做了监听），generation_called=false；
文字路径无证据仍是 model_called=false。引用只表示提供的证据，不等于逐句事实校验。
ASR、检索错误及等待超时不会释放模型说话；旧 Future 不会注入下一请求。
取消可撤销尚未开始的任务，已运行的 CUDA generate 不能安全强杀，关闭 worker 会等待其结束再释放权重。
CLI 退出会依次关闭后台 worker、MiniCPM 和日志。

## 检索和接口

rag_core.py 迁移原项目的字符2–4 gram TF-IDF、余弦相似度、证据卡及引用结构。
RAGRetriever 检查知识 JSON、重复 ID，并在内容 SHA256 改变后重建索引。
默认 `--top-k 3 --min-score 0.08 --min-shared-bigrams 2 --max-evidence-chars 900`。
词面检索有误召回/漏召回和证据截断风险；分数不是置信概率，无语义重排或多轮问题改写。

- `ASRRetrievalWorker(LocalWhisperASR(...), retriever)`：后台 ASR 和检索，需 close。
- `AsyncAudioRAGPipeline(worker, runtime).answer(audio, on_chunk=..., on_event=...)`：文件音频并行入口。
- `OfficialDuplexRuntime.stream_pending_answer(...)`：等待 Future，同时维持 AUDIO 监听及注入门控。
- `RAGPipeline.answer(query)`：文字入口；底层现在也只用 AUDIO+text_list。
- `OfficialDuplexRuntime.stream_answer(...)`：已知问题和证据的 AUDIO 注入与生成。

API 调用者应 finally 先 worker.close() 再 runtime.close()，避免模型清理与后台推理交叉。
复用示例见 run_rag.py 和 tests/integration_async_audio.py。

输出含 results.json、turn_*.json、events.jsonl，以及实际生成的24kHz WAV。
异步结果额外保存 asr、background 的 monotonic 时间、runtime_events；
事件包含 backend.submitted、audio.listen、asr.final、retrieval.result、rag.injected、answer.chunk。
`mode` 字段与运行时检查用于确认全程 AUDIO。时间区间重叠不能替代 CUDA profiler 的 kernel 级测量。
正常/无证据退出码0，回答不完整2，异常1。

## 验证

```bash
python -m pytest tests test_evaluation.py -q

# 真实同卡并行 + 人为延迟检索，用于验证等待门控，保留新输出目录。
CUDA_VISIBLE_DEVICES=0 python -u tests/integration_async_audio.py \
  --audio /path/to/question.wav

# 全AUDIO文字多轮热更新回归，指定未使用过的输出目录。
CUDA_VISIBLE_DEVICES=0 python -u tests/integration_rag.py \
  --output outputs/new_text_regression
```

最新验证见 ASYNC_AUDIO_RAG_REPORT.md。RAG_REPORT.md 及 AUDIO_*_REPORT.md 保留历史实验，
其中串行 ASR、独立 TEXT、音频空回答等描述对应旧实现，不代表当前调用顺序。

本仓库不包含 model/、models/、outputs/。报告中的原工作区路径是历史记录；新环境以本指南和 README 的命令为准。
