# 同卡异步 ASR/RAG 与全程 AUDIO 实现验证

日期：2026-11-16。修改在 minicpm_official_rag_test 下，旧 minicpm_rag_test 和官方模型文件未修改。
复用现有 Python 环境、MiniCPM-o 4.5 与 Whisper-small 权重，无新依赖、无下载。
本次所有真实验证使用物理 GPU 0，MiniCPM BF16 和 Whisper FP16 常驻同一逻辑 cuda:0。
TF-IDF 检索为后台 CPU 计算，不需要第二张 GPU。

## 实现

- async_audio_rag.py：LocalWhisperASR 常驻一个后台线程和独立 CUDA stream；ASRRetrievalWorker
  执行依赖顺序 ASR→检索，通过 Future 交回 PreparedAudioQuery。MiniCPM 与 ASR/检索并行。
- official_runtime.py：每次 prefill 强制要求非空 audio_waveform，检查官方 current_mode=AUDIO；
  后台未完成时，原音频或静音 unit 强制监听；结果返回后，静音+text_list 在 AUDIO 下合并，
  注入所在 unit 仍强制监听。下一 AUDIO forward 才释放监听，生成正常回答。
- 同步文字入口也改为 AUDIO+text_list，不再单独输入 TEXT。prompt 末尾固定一个换行。
- run_rag.py：--audio 默认走上述并行路径；--device cuda:0 默认同时指定两个模型。
  --asr-device 如显式设置必须一致。仅检索入口不加载 MiniCPM。
- 后台从不访问 MiniCPM 会话；同一 runtime 和 pipeline 单一 owner；错误/超时不放行说话，
  陈旧 Future 不会注入下一 turn。关闭时先等待后台结束并释放 Whisper，再关闭 MiniCPM。

## 启动时模型构造的并发问题

首次将两个 from_pretrained 同时运行，出现 Cannot copy out of meta tensor 错误，结果保留在
outputs/async_audio_rag/results.json。此时仅构造失败，没有伪装为推理成功。
修正为启动阶段顺序构造两个模型，推理阶段仍并行。新增测试强制禁止两个构造过程重叠，
修正后真实运行通过。不能将启动顺序加载误解为每轮串行 ASR/RAG。

## 真实 CLI：后台并行

命令：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u minicpm_official_rag_test/run_rag.py \
  --audio minicpm_demo/tests/cases/common/user_audio/当出现植物大战僵尸的时候提醒我.wav \
  --output minicpm_official_rag_test/outputs/async_audio_rag_v2
```

输出记录：outputs/async_audio_rag_v2/results.json 与 events.jsonl。

- 实际 ASR：“当出现植物大战僵尸的时候提醒我”，设备 cuda:0，命中 plant-zombies。
- ASR 后台任务区间约0.5812秒，其中约0.5805秒与 MiniCPM AUDIO unit 的主机调用区间重叠。
- 随后的检索区间约0.00576秒，也与第二个 AUDIO unit 处理区间重叠。
- 每个监听事件和注入均为 AUDIO；注入事件 forced_listen=true，回答逐 unit mode=AUDIO。
- 回答正常结束，文字：“好的，当出现植物大战僵尸的时候我会提醒你。”
- 这条自然时序下 ASR/检索在原音频输入完成前已返回，无额外等待 unit。
- 该回答未包含资料的特有细节，不能视为 RAG 事实利用质量通过；status=answered 仅表示
  模型正常结束且有非空文本，没有自动判定逐句事实性。

上述重叠为 perf_counter 单调时钟的主机区间观测，不是 CUDA profiler 的 kernel 级并发证明。
独立 CUDA stream 允许同卡调度，不保证 GPU 上每个 kernel 同时执行或延迟低于串行。

## 真实模型：检索延迟与监听门控

脚本 tests/integration_async_audio.py 使用真实 ASR、真实 MiniCPM、真实检索。
检索返回由测试事件延迟到两个 waiting AUDIO unit 完成，用于受控验证门控，不能当性能数据。

输出：outputs/async_audio_gate_test/results.json，status=passed。

- ASR 与 MiniCPM AUDIO 主机调用区间重叠约0.6298秒。
- 检索等待阶段两个 AUDIO unit 全部强制 listen，注入所在 unit 同样强制 listen。
- 注入开始时间晚于检索完成时间，回答回调在注入事件之后才发生。
- ASR worker 与 MiniCPM owner 是不同线程；两个模型配置为同一 cuda:0。
- 模型回答：“当出现植物大战僵尸的时候，提醒您检查阳光、坚果冷却时间和草坪状态。”
- 此例生成全部目标细节；与自然时序组的差异说明格式/时序仍影响回答，不能据此声称所有输入稳定。

## 文字多轮回归

tests/integration_rag.py 新增 --output 参数，保留旧输出不覆盖。
本次新结果在 outputs/audio_mode_text_regression/results.json：

1. 原知识：“根据差旅报销制度，上海出差住宿可报销每晚六百元。”
2. 同一会话修改测试知识副本：“根据差旅报销制度，上海出差住宿可报销每晚四百三十七元。”
3. 量子纠缠问题：应用层知识不足拒答，无模型调用、无 WAV。

三轮断言全部通过。437为测试更新，不是真实政策。底层本次已使用 AUDIO 混合注入而非旧 TEXT 路径。

## WAV 独立转写

对自然并行、延迟门控、两条差旅回答的实际 WAV 独立运行 Whisper-small，没有提供答案提示。
汇总输出：outputs/async_audio_validation/asr_results.json。

- 自然并行：“好的 當出現植物大戰殭屍的時候會提醒你”。
- 延迟门控：“當出現植物大戰相失的時候提醒您檢查陽光,監國冷卻時間和草坪狀態”。
- 差旅原知识：“根據查理報銷制度,上海出差住宿可報銷每碗600元。”
- 差旅更新：“人據《差旅報銷制度》上海出差住宿可報銷每晚437元”。

金额和主要资料细节可识别，但存在同音字错误；这不是人工音质验收。

## 自动化检查与范围

31项测试通过。新增覆盖真正的后台与模型调用交叠、单一 MiniCPM owner、同卡参数、
全程 AUDIO、注入期间监听、下一 AUDIO unit 放行、超时、ASR异常、无证据、回调异常后复用、
无效音频提前拒绝，以及启动构造不能并发。所有模块编译检查通过。

官方 modeling_minicpmo.py SHA256 保持：
25359f891d1aa81de16b51a24a97b8c8262b83ce3920c36b92be441fde8e4282。

当前入口仍是音频文件 CLI：后台使用完整已读文件做 ASR，MiniCPM 同时分块处理该音频。
没有新增实时麦克风采集、partial ASR、VAD触发、WebSocket、用户打断或严格实时帧调度。
不把文件回放的处理重叠宣称为这些服务已经实现。模型输出对注入格式/时序的敏感性仍存在。

## Qwen3-ASR vLLM 真实服务验证（2026-11-17）

后续已在独立环境完成 Qwen3-ASR-1.7B 的真实 vLLM 推理：

- vLLM 0.14.0 + Torch 2.9.1+cu128 + Transformers 4.57.6。
- 官方 `/api/start`、`/api/chunk`、`/api/finish` 流式服务可用。
- 13.7 秒示例音频 final transcript 为
  “当出现《植物大战僵尸》的时候，提醒我。”，vLLM 推理约 0.93 秒。
- `Qwen3ASRHTTPBackend` 已接入 `Qwen3ASRStreaming`；final transcript 命中
  `plant-zombies`，并在测试 runtime 中完成
  `asr.final → retrieval.result → rag.injected`。
- 注入使用 AUDIO `audio_waveform` + `text_list`，注入 unit 保持 `force_listen`，
  注入文本包含证据并保留末尾单个换行。

vLLM 0.11 缺少 Qwen3-ASR 自定义架构，0.13 缺少 0.14 backend 模块；
0.14 与 Torch 2.8 组合会因二进制 ABI 不匹配在加载 `_C` 时触发
`std::bad_alloc`。可运行环境、权重和独立验证入口见
[QWEN3ASR_ENVIRONMENT.md](QWEN3ASR_ENVIRONMENT.md)。
