# MiniCPM 流式语音 RAG 人工评测控制台设计

日期：2026-11-17

## 意图与原因

为 `knowledge/squad_zh_25.json` 提供一个人工可操作、可展示的本地控制台。
使用者按住麦克风按钮说话，浏览器持续发送 16 kHz 单声道 float32 PCM；
服务端同时驱动 Qwen3-ASR、TF-IDF 检索和 MiniCPM AUDIO 监听，并在用户松开按钮后
完成 final ASR、证据注入、文本生成和 24 kHz PCM 音频输出。

控制台核心是 RAG 运行观测台。需要同时看到识别文本、检索命中、证据内容、注入状态、
模型生成状态和各阶段时延。回答只是结果之一，不是唯一展示目标。

音频交互参考 `avatar_full/audio_dual_avatar`：

- 浏览器 `AudioWorklet` 直接采集 PCM。
- WebSocket 持续传输控制消息和 PCM 音频块。
- 输出 PCM 在浏览器用 Web Audio API 排程播放。
- 不使用 `MediaRecorder`。
- 不把麦克风输入编码成 WAV、WebM 或 Opus 文件。
- 不在服务端或浏览器持久化麦克风音频。

## 目标

- 按住按钮从麦克风连续发送 PCM，松开按钮结束一个问题。
- MiniCPM 在用户说话期间持续接收 AUDIO unit，而不是等整段录音结束后才开始。
- Qwen3-ASR 随每 1 秒 PCM 块产生 partial，并在结束后返回 final transcript。
- MiniCPM AUDIO 监听、ASR worker 和 RAG worker 使用独立执行通道异步运行。
- partial transcript 异步触发预检索；final transcript 触发权威检索并用于证据注入。
- 按事件增量显示 ASR partial、final、检索命中、证据、AUDIO 注入和回答。
- 展示 MiniCPM 返回的 24 kHz PCM 增量，并在浏览器内存中连续播放。
- 展示每条命中的文档 ID、标题、来源、分数和证据正文。
- 以统一时间线展示本轮阶段时延，并保留最近若干轮用于人工比较。
- 默认使用 `knowledge/squad_zh_25.json` 和全文检索。
- 默认把每一轮视为独立 benchmark 问题，开始新一轮时重置 MiniCPM 会话。
- 第一版控制台只接入 Qwen3-ASR HTTP 流式后端；Whisper 继续由现有 CLI 和测试使用。
- 无证据、超时、ASR 错误和服务不可用时给出明确状态。
- 保持现有的单 owner、串行会话和 AUDIO 注入约束。

## 非目标

- 不实现 VAD 自动断句；第一版由“按住说话、松开结束”显式确定 turn 边界。
- 不实现用户打断模型回答。后续可参考 avatar 的 `input.interrupt` 增加。
- 不把麦克风 PCM 保存为任何媒体文件。
- 不提供回答历史音频文件下载；页面刷新后上一轮音频不可恢复。
- 不替换现有字符 n-gram TF-IDF 检索，也不增加向量检索或重排。
- 不自动判断回答的事实正确性，只展示证据和模型输出供人工判断。
- 不实现多用户、登录、数据库或公网部署。
- 不自动生成 `squad_zh_25.json` 中问题的音频。

## 反目标

- 浏览器使用 `MediaRecorder` 录制后上传完整音频文件。
- 服务器先把麦克风 PCM 写盘，再调用现有文件式 `run_rag.py`。
- 每轮启动子进程并重复加载 MiniCPM。
- 用户仍在说话时让 MiniCPM 只做无效等待，错过真实 AUDIO 输入。
- 只展示最终回答，隐藏 ASR、检索、证据和注入状态。
- 把说话时长混入 ASR 服务时延，造成指标不可比。
- 在没有证据时仍显示成正常 RAG 命中。

## 约束

- MiniCPM 主环境仍为 Torch 2.8.0+cu128、Transformers 4.51.0。
- Qwen3-ASR 必须继续在独立 `.venv-qwen` 中通过 HTTP 服务运行。
- MiniCPM 和 Whisper 必须使用同一个逻辑 CUDA 设备；Qwen HTTP 服务可使用另一张 GPU。
- MiniCPM 会话必须串行访问，一次只允许一个 active turn。
- 第一版单个问题最长 30 秒，达到上限自动发送 `input.end`。
- 浏览器麦克风需要安全上下文，即 `localhost` 或 HTTPS。
- Qwen HTTP 每个 1 秒 PCM 块调用仍会产生网络和推理开销，各输入队列必须有界。

## 方案选择

### 方案 A：Gradio 麦克风控件

开发快，但输入组件通常以完整音频文件为中心，不适合持续 PCM 流、逐块 partial、
阶段事件和自定义时延观测。没有采用。

### 方案 B：FastAPI + 流式 NDJSON + 录音文件上传

实现简单，但录音完成后才提交，无法让 MiniCPM 在用户说话时并行监听，
也违背不保存麦克风音频的要求。没有采用。

### 方案 C：FastAPI WebSocket + AudioWorklet PCM 双向流

浏览器持续发送 16 kHz float32 PCM；服务端通过同一 WebSocket 返回控制事件、
文本增量、ASR partial、检索结果和 24 kHz float32 PCM；浏览器用 AudioContext 排程播放。
这与 `audio_dual_avatar` 的成熟音频链路一致，也能自然表达状态和时延。采用此方案。

界面采用已确认的 A“运行控制台”布局，把 B 的阶段时间线放在底部作为可展开面板。

## 用户界面

### 顶部工具栏

- 页面状态：模型加载中、空闲、采集音频、等待 ASR、检索中、注入中、生成中、错误。
- 当前知识文件，默认 `knowledge/squad_zh_25.json`。
- 检索模式：全文（默认）或仅标题。
- `top-k`、`min-score`、最大证据长度。
- ASR 后端和健康状态：第一版固定为 `qwen-http`。
- 会话重置按钮。

知识文件在服务启动时通过 `--knowledge` 固定，前端只显示文件名和 SHA256 摘要。
选择另一份知识库需要重启服务。

检索参数只影响下一轮检索，不触发 MiniCPM 重新加载。第一版控制台固定使用
`qwen-http`，因为 Whisper 当前没有实时 partial，不能提供同样的流式 ASR 观测。
Whisper 仍保留在命令行文件音频入口和自动化测试中。

### 左侧链路状态

按顺序显示：

1. 麦克风采集
2. 流式 ASR
3. final transcript
4. TF-IDF 检索
5. AUDIO 证据注入
6. MiniCPM 文本生成
7. MiniCPM 音频生成

每个阶段显示等待、进行中、完成、失败及耗时。ASR 阶段展开显示 partial 文本。
MiniCPM AUDIO 监听和 ASR 共享同一时间轴，应明确显示它们并行发生。

无证据沿用当前音频应用路径：`retrieval.result` 中 `usable=false`，随后仍把空
reference 注入 MiniCPM，允许模型决定回答或拒答。界面必须显示“无证据”，
但不能把该轮当作 `turn.error`。

### 中间问答区

- 麦克风电平波形和已说话时长。
- “按住说话 / 松开结束”按钮。
- ASR partial 和 final transcript。
- 当前问题和 MiniCPM 回答文本。
- 回答 PCM 在浏览器内存中连续播放的状态。
- 首段文本时延、首段音频时延和总时延。
- 文字输入仅作为无麦克风权限时的备用调试入口；它也走现有 AUDIO 注入路径。

### 右侧证据区

- 按分数显示 top-k 命中。
- 每条显示标题、文档 ID、来源、分数、前 200-300 字摘要。
- 展开后显示实际送入 MiniCPM 的完整证据卡。
- 显示检索是否 usable 以及无证据状态。
- 至少展示分数和文档 ID，双字词交集作为可选诊断信息。

### 底部时延追踪

默认折叠，展开后显示本轮水平时间线：

- 浏览器首块 PCM
- 首个 ASR partial
- ASR final
- 检索
- 等待注入
- AUDIO 注入
- MiniCPM 首个文本 delta
- MiniCPM 首个音频 delta
- 回答完成

下方显示最近 10 轮的关键指标表格。每个指标标明计时起点，例如
“首个 PCM 到达后 624 ms”，避免把说话时长混入服务端 ASR 时延。

## 音频协议

连接：

```text
ws://host/ws/console
wss://host/ws/console
```

客户端到服务端：

```json
{
  "type": "session.start"
}
```

```json
{
  "type": "input.start",
  "sample_rate": 16000,
  "format": "pcm_f32le_16000",
  "turn_id": "client-generated-id"
}
```

```json
{
  "type": "input.audio.append",
  "sequence": 0,
  "audio": "<base64 float32 PCM>"
}
```

```json
{
  "type": "input.end",
  "sequence": 41,
  "reason": "button_release"
}
```

```json
{
  "type": "session.reset"
}
```

服务端到客户端：

- `session.ready`
- `turn.started`
- `input.ack`
- `asr.partial`
- `asr.final`
- `retrieval.result`
- `audio.listen`
- `rag.injected`
- `answer.text.delta`
- `answer.audio.delta`
- `answer.done`
- `turn.error`
- `session.resetting`

`answer.audio.delta`：

```json
{
  "type": "answer.audio.delta",
  "sample_rate": 24000,
  "format": "pcm_f32le_24000",
  "unit": 0,
  "audio": "<base64 float32 PCM>"
}
```

base64 仅用于 WebSocket JSON 传输，不形成音频文件，不写入日志。首个实现保持与
`audio_dual_avatar` 一致；如果后续测出 base64 开销明显，可升级为带定长 header 的
binary WebSocket frame，而不改变业务事件。

## 浏览器音频实现

### 采集

- `getUserMedia` 请求 mono 音频，优先启用 echo cancellation。
- 创建 `AudioContext({sampleRate: 16000})`。
- 加载独立 `capture-worklet.js`。
- AudioWorklet 累积恰好 1 秒、即 16000 个 float32 样本后，通过 transferable
  `Float32Array` 发回主线程。
- 如果上下文实际采样率不是 16000，使用线性插值降到 16 kHz。
- 主线程把 PCM 转成 base64，并在 WebSocket 打开时发送。
- 主线程不拼接完整问题音频，不创建 Blob、Object URL、WAV 文件或下载链接。
- 使用 `AnalyserNode` 绘制实时电平和说话时长。
- 松开按钮时，要求 worklet 发送最后一个不足 1 秒的块，然后发送 `input.end`。
- 30 秒上限由前端定时器和后端正向验证共同保证。

### 播放

- 收到 `answer.audio.delta` 后把 base64 解码为 float32。
- 每个增量创建一个 24 kHz `AudioBuffer`。
- 使用 `AudioContext.currentTime` 维护 `next_play_time`，以约 80-120 ms 初始缓冲排程播放。
- 后续 chunk 接在上一个 `AudioBufferSourceNode` 的结束时间之后，避免按收到时间粗暴播放。
- 浏览器内存仅保存本轮最多 30 秒的 24 kHz PCM，用于本轮重播；页面刷新后丢弃。
- 不调用 `decodeAudioData` 解析 WAV，因为服务端不会输出 WAV。
- 支持停止播放和重新播放内存中的本轮音频。

## 服务端音频与 RAG 数据流

```text
AudioWorklet (float32 PCM 16k)
  → WebSocket input.audio.append
  → ConsoleTurnManager
      ├─ MiniCPM owner queue → 1-second AUDIO force-listen
      └─ ASR worker queue → Qwen streaming /api/chunk
                              ├─ asr.partial
                              │    └─ RAG worker queue → retrieval.partial
                              └─ asr.final
                                   └─ RAG worker queue → retrieval.final
  → button release (input.end)
      ├─ flush final short PCM with zero padding → MiniCPM listen
      ├─ Qwen /api/finish → final transcript
      └─ authoritative RAGRetriever.retrieve
  → coordinator waits for all PCM units + final retrieval
  → silence AUDIO + text_list evidence injection
  → MiniCPM generate
      ├─ text delta → answer.text.delta
      └─ 24 kHz PCM → answer.audio.delta
  → answer.done + normalized latency metrics
```

服务端不在磁盘上保存输入 PCM 或输出 PCM。console 结果日志只保存文本、分数、
事件时间和音频样本数，不保存音频字节或媒体 URL。

## 后端结构

建议新增：

- `console_server.py`：FastAPI、WebSocket、静态文件和 CLI 生命周期。
- `console_runtime.py`：单个 active turn、PCM 队列、事件归一化和时延计算。
- `streaming_audio_rag.py`：PCM 流式 ASR、独立 RAG worker、MiniCPM 实时监听、
  final RAG 注入和生成编排。
- `static/console/index.html`：控制台结构。
- `static/console/console.css`：视觉样式和响应式布局。
- `static/console/console.js`：WebSocket 状态机、PCM 发送和文本展示。
- `static/console/audio-stream.js`：AudioWorklet 采集、重采样和 PCM 播放队列。
- `static/console/capture-worklet.js`：AudioWorkletProcessor。
- `tests/test_streaming_audio_rag.py`：流式 ASR、队列、注入和时延测试。
- `tests/test_console_server.py`：WebSocket 协议和控制台 API 契约测试。

复用：

- `RAGRetriever` / `RetrievalConfig`
- `Qwen3ASRHTTPBackend`
- `OfficialDuplexRuntime` 的 AUDIO prefill、listen、inject 和 generate 能力
- 现有 `audio_dual_avatar` 已验证的 AudioWorklet 和 PCM 播放思路

需要新增一个真正的流式编排入口，不能直接把完整 `numpy` 音频数组传给现有
`AsyncAudioRAGPipeline.answer`。现有 `ASRRetrievalWorker` 把 ASR 和检索串在同一个
executor 中，不能用于控制台；控制台需要拆成独立 ASR worker 和独立 retrieval worker。
现有文件式入口继续保留给 CLI 和测试使用。

## 三路异步执行

### MiniCPM AUDIO owner

- 独占 MiniCPM session、`streaming_prefill` 和 `streaming_generate`。
- 只消费 1 秒 AUDIO unit 队列。
- 用户说话时持续 force-listen，不回退到 TEXT。
- ASR/RAG 慢于 AUDIO 时，继续用静音 AUDIO unit 监听。
- 不等待 ASR HTTP 请求或 TF-IDF 检索完成后才处理已经到达的音频。

### ASR worker

- 独立线程和独立有界队列。
- 每个 1 秒 PCM unit 立即调用 Qwen `/api/chunk`。
- partial 更新发布为带 monotonic revision 的事件。
- `input.end` 后调用 `/api/finish` 并发布 final transcript。
- ASR worker 不执行 RAG，不持有 MiniCPM owner 锁。

### RAG worker

- 独立线程和独立有界队列。
- 收到非空 partial 后执行标签为 `partial` 的预检索，结果发布为
  `retrieval.partial`，只用于展示预热效果和估算时延。
- revision 单调递增；队列使用 latest-wins，积压时丢弃尚未开始的旧 partial 检索。
- final transcript 到达后提交更高优先级的 `final` 检索。
- 文本与最后一次预检索相同时可以直接复用结果；不同则重新执行最终检索。
- `retrieval.final` 是唯一允许进入 MiniCPM 证据注入的结果。
- 旧 revision 或上一 turn 的检索结果必须丢弃，不能覆盖当前状态。

### Coordinator

- 监听 ASR final queue 和 final retrieval queue。
- 等待两个依赖同时满足：全部输入 AUDIO unit 已被 MiniCPM 消费，且
  `retrieval.final` 已完成。
- 然后注入证据并启动生成。
- 任一 worker 异常时取消当前 turn 并释放队列，不能让旧任务进入下一轮。

## MiniCPM 实时监听

- 浏览器每 1 秒发送一个 PCM 块。
- 服务端把同一个 PCM unit 同时扇出给 MiniCPM owner queue 和 ASR worker queue。
- AUDIO owner 在输入期间持续执行 force-listen，不回退到 TEXT。
- ASR worker 独立消费同一份只读 PCM 数据并调用 Qwen `/api/chunk`。
- 用户松开按钮时，最后一个不足 1 秒的 unit 用零填充到 1 秒。
- AUDIO owner 处理完所有已提交 unit 后，继续静音监听，等待 final ASR 和检索。
- 检索结果到达后，以静音 AUDIO + `text_list` 注入证据；注入 unit 保持 force-listen。
- 下一个 AUDIO unit 释放监听并进入生成。
- 输入队列有上限；积压时返回明确 `backpressure` 错误，而不是无限占用内存。

## 会话与并发

- 模型启动后后台按顺序 warmup ASR 和 MiniCPM。
- warmup 完成前 WebSocket 返回 `session.loading`，麦克风按钮禁用。
- 默认每一个问题独立：`input.start` 前重置 MiniCPM session，避免 URObench 问题互相污染。
- 第二个 active turn 被拒绝，并返回明确 `turn_busy`。
- 前端在等待回答完成期间禁用麦克风按钮。
- 模型生成期间禁用“重置会话”按钮；第一版不实现生成中途取消。
- 服务端关闭时先停止 ASR worker，再关闭 MiniCPM。
- 客户端断开 WebSocket 时，释放当前 PCM 队列和 ASR session；不能留下后台 turn。

## 时延定义

所有服务端时间使用 `time.perf_counter()`，以服务端收到首个非空 PCM 块为统一零点。

- `input_duration_ms`：已接收 PCM 样本数 / 16000，属于用户说话时长，不纳入 ASR 耗时。
- `first_asr_partial_ms`：首个非空 partial 到达时间。
- `first_partial_retrieval_start_ms`：首个 partial 预检索开始时间。
- `first_partial_retrieval_done_ms`：首个 partial 预检索完成时间。
- `asr_final_ms`：`input.end` 到 final transcript 的时间。
- `retrieval_final_ms`：final transcript 完成到权威检索结果为可用/不可用的时间。
- `wait_to_inject_ms`：检索完成到注入开始。
- `injection_ms`：注入事件 `finished_s - started_s`。
- `first_text_delta_ms`：注入完成到首个非空文本 delta。
- `first_audio_delta_ms`：注入完成到首个 24 kHz 音频 delta。
- `answer_total_ms`：首个 PCM 到 `answer.done`。
- `unit_generation_ms[]`：MiniCPM 每个 unit 的现有 generation 指标。
- `client_playback_lead_ms`：浏览器初始播放缓冲，由浏览器单独记录。
- `asr_minicpm_overlap_ms`：ASR worker 活跃区间与 MiniCPM AUDIO unit 处理区间的重叠。
- `asr_rag_overlap_ms`：ASR partial 生成期间与预检索执行区间的重叠。

界面同时显示“说话时长”和“松开后处理时长”，避免把用户说话时间误认为模型延迟。

## 无证据与失败行为

- 无证据不是 `turn.error`，UI 标记“检索无证据”，继续展示模型行为。
- final transcript 为空时视为可恢复的“未检测到语音”，不调用 MiniCPM，
  清空本轮 PCM 和 ASR session，允许用户直接重新按住说话。
- Qwen 不可用时，本轮在 `input.start` 明确失败，不启动 MiniCPM 生成。
- 队列积压、非法 PCM、超过 30 秒和 WebSocket 协议错误返回结构化错误。
- MiniCPM 生成中断时清理 active session；下一轮默认重新 prepare。
- 浏览器断开后不尝试保存未完成音频。

## 安全与输入校验

- 默认只监听 `127.0.0.1`；远程访问使用 HTTPS 或 SSH 端口转发，保证麦克风权限可用。
- WebSocket 只接受约定的 JSON 消息类型。
- PCM 必须是有限 float32、mono、16 kHz；单块和整轮样本数都有限制。
- 知识文件只允许来自项目 `knowledge/` 目录的白名单。
- 不把任意文件路径、命令行参数、ASR 文本或模型文本直接拼入 HTML。
- 输入和输出 PCM 不写盘；日志不含音频字节或 base64。
- JSON 指标日志使用 UTC turn ID，不包含媒体内容。

## 测试策略

### 单元测试

- AudioWorklet 输出块顺序、最终 flush、1 秒分块和重采样。
- PCM base64 编解码后保持 float32 数值。
- 有界输入队列和 backpressure。
- Qwen partial 回调只转发非空更新并保持 revision 顺序。
- 同一 1 秒 PCM unit 被 MiniCPM 和 ASR 两个队列同时消费。
- partial 预检索 latest-wins，旧 revision 结果不覆盖新 revision。
- final retrieval 优先于尚未开始的 partial retrieval。
- 最后不足 1 秒的输入块零填充后送入 MiniCPM。
- 事件时延归一化，说话时长与服务端处理时长分离。
- 默认逐题 session reset、busy 拒绝和断开清理。
- 无证据状态不映射成 `turn.error`。
- 空 transcript 返回 `no_speech` 且不调用 MiniCPM。

### 集成测试

- 用测试客户端通过 WebSocket 发送合成 PCM，验证事件顺序。
- 用现有植物大战僵尸 WAV 在测试进程内读成 PCM、再按块发送；测试文件不进入 console 存储。
- Qwen HTTP 可用时验证 partial → final → retrieval → injection → text/audio delta。
- 用阻塞 fake ASR 验证 MiniCPM AUDIO unit 仍能继续消费；用慢 RAG 验证 ASR partial
  仍能继续到达。
- 用受控延迟 fake retrieval 验证 partial 检索和 final 检索异步、final 结果权威。
- 用 `squad_zh_25.json` 验证全文检索命中对应上下文。
- 真实 GPU 验证模型只加载一次、连续两轮自动 reset 但不重复加载。
- 断言 console 运行结束后没有新增输入/输出音频文件。

### 人工验收

1. 启动 Qwen ASR 服务和 console。
2. 浏览器允许麦克风。
3. 按住按钮说出一个 `squad_zh_25.json` 相关问题，观察实时电平和 partial。
4. 松开按钮后看到 final transcript、命中证据和分数。
5. 看到 AUDIO 注入状态，然后文本和音频增量开始出现。
6. 回答音频连续播放，时间线显示各阶段真实时延。
7. 确认浏览器下载区、服务端输出目录和工作区均没有新增麦克风音频文件。
8. 连续提两轮，确认逐题会话隔离且模型没有重复加载。

## 实施顺序

1. 抽取或参考 `audio_dual_avatar` 的 AudioWorklet PCM 采集和播放代码。
2. 增加 Qwen 流式 PCM append 和 partial 回调。
3. 拆分独立 ASR worker 和 latest-wins RAG worker。
4. 增加 MiniCPM 实时 AUDIO 队列、三路并发和 final 注入 coordinator。
5. 实现 FastAPI WebSocket turn manager。
6. 实现控制台布局、证据面板和时延时间线。
7. 完成单元、WebSocket 集成和真实 GPU 验收。

## 留给实现者的决定

- FastAPI/uvicorn 的精确版本和启动参数。
- partial 回调通过 worker 参数还是缩小版 progress emitter 传递。
- 输出播放缓冲的具体延迟，默认从 100 ms 验证。
- 是否后续升级为 binary WebSocket frame，第一版与参考实现一致使用 JSON + base64。
