# AUDIO 门控与 RAG 文本末尾换行对照

日期：2026-11-16。GPU 0，使用现有本地官方模型，无下载、无模型源码修改。
实验脚本：`probe_audio_gate_newline.py`；原始输出：`outputs/audio_gate_newline/results.json`。

## 实验控制

真人音频由本地 Whisper-small 转写，实际 query 为“当出现植物大战僵尸的时候提醒我”。检索命中 plant-zombies，证据要求检查阳光、坚果冷却时间和草坪状态。
各组独立 reset_session/prepare 并 prime 一秒监听；同一 seed=42、系统提示词、参考音色、知识证据及问题。
原音频按一秒分块（末段补零），全部强制监听；然后输入两个一秒静音监听 unit，模拟检索等待。
检索实际上提前完成，不测真实异步检索延迟。回答为 greedy，最多24 units，没有强制 speak。
末尾换行指 `build_rag_prompt(query,evidence) + "\n" * n`，即完整 prompt 末尾，未改变 reference 内部格式。

- text_release：等待后独立 text_list 注入，释放监听并生成。
- audio_release：等待后静音 audio_waveform 与 text_list 同次注入，生成前释放监听。
- audio_hold：同上，但注入所在 unit 也强制监听；下一静音 AUDIO unit 才释放监听。
- audio_no_evidence：相同等待过程后仅注入静音，没有 RAG 或问题文本。

## 全部结果

| 路径 | 末尾换行数 | 回答 | 证据关键词命中 | 自然结束 |
|---|---:|---|---|---|
| text_release | 0 | 空 | 无 | True |
| text_release | 1 | 好的，当出现植物大战僵尸的时候我会提醒你的。 | 无 | True |
| text_release | 2 | 当出现植物大战僵尸的时候，提醒您检查阳光、坚果冷却时间和草坪状态。 | 阳光、坚果、冷却、草坪 | True |
| audio_release | 0 | 好的，当出现植物大战僵尸的时候我会提醒你。 | 无 | True |
| audio_release | 1 | 当出现植物大战僵尸的时候，提醒您检查阳光、坚果冷却时间和草坪状态。 | 阳光、坚果、冷却、草坪 | True |
| audio_release | 2 | 好的，当出现植物大战僵尸的时候我会提醒你。 | 无 | True |
| audio_hold | 0 | 好的，当出现植物大战僵尸的时候我会提醒你。 | 无 | True |
| audio_hold | 1 | 当出现植物大战僵尸的时候，提醒您检查阳光、坚果冷却时间和草坪状态。 | 阳光、坚果、冷却、草坪 | True |
| audio_hold | 2 | 好的，当出现植物大战僵尸的时候我会提醒你。 | 无 | True |
| audio_no_evidence | 0 | 空 | 无 | False |

audio_no_evidence 组24个生成 unit 全为 listen，达到上限，不能记作自然完成或空文本 EOT。

## 首步分布

以下为原始 logits 的 softmax，未经过 decoder 的 forbidden-token mask、重复惩罚或采样过滤，不应当作最终采样概率。实际解码为 greedy，不是随机抽样频率。

| 路径 | 换行数 | pending 的 EOT 概率 | 最后文本 feed 的 EOT 概率 | pending 的 top token |
|---|---:|---:|---:|---|
| text_release | 0 | 0.702421 | 0.702421 | `<|turn_eos|>` |
| text_release | 1 | 0.0635271 | 0.0635271 | `<think>` |
| text_release | 2 | 0.412136 | 0.412136 | `<|chunk_eos|>` |
| audio_release | 0 | 1.35961e-10 | 0.755233 | `<|listen|>` |
| audio_release | 1 | 1.35961e-10 | 0.0382335 | `<|listen|>` |
| audio_release | 2 | 1.35961e-10 | 0.484382 | `<|listen|>` |
| audio_hold | 0 | 1.35961e-10 | 0.755233 | `<|listen|>` |
| audio_hold | 1 | 1.35961e-10 | 0.0382335 | `<|listen|>` |
| audio_hold | 2 | 1.35961e-10 | 0.484382 | `<|listen|>` |

混合 AUDIO 分支的 pending 来自文本之前的音频阶段，因此三种换行的 pending 完全一致；
文本 feed 后的诊断分布随换行变化，但本实验没有将它覆盖回 pending_logits。
证据和换行仍改变 KV，并影响后续 AUDIO unit 的生成。audio_hold 组的释放时分布另见原始 JSON。

## 结论

1. 本例验证了完整 AUDIO 监听门控：等待期间与注入 unit 均未说话，下一 AUDIO unit 释放后可生成证据细节。
2. 末尾一个换行在两条 AUDIO 路径中均生成全部目标细节；零或两个换行仅泛化应答。不能将追加换行视为普适修复。
3. 同样存在 AUDIO→TEXT 切换，两个换行的 text_release 组仍能回答且包含全部目标细节，反驳“一旦切换必然 EOT”的绝对解释。
4. 训练时是否覆盖混合输入或模式切换，不能从本次推理对照确定。更谨慎的判断是输入格式和音频上下文会显著影响控制 token 分布。
5. 单条音频、单份知识、单 seed、greedy 结果不构成泛化或实时部署验证；关键词命中不等同于完整事实性评测。
6. 正式 official_runtime.py 尚未改为此实验策略；本次任务交付对照脚本和记录，不掩盖旧入口限制。

## 独立音频回转与检查

已使用本地 Whisper-small 对实际生成 WAV 转写，没有输入预期答案提示，记录见
outputs/audio_gate_newline/asr_results.json。audio_release_nl1 和 audio_hold_nl1 的回转结果一致：
“當出現植物大戰相失的時候提醒您檢查陽光,監國冷卻時間和草坪狀態”。
主要细节可识别，但“僵尸”“坚果”存在同音误差；不据此宣称音质完全通过。
text_release_nl1 的回转明显不完整且误识别，不能把该组文字输出视作语音质量通过。
空文本的短音频再次被 ASR 转写成“謝謝”，不能当作有效回答。

原22项自动化测试仍通过；本次另检查所有等待 unit 和 audio_hold 注入 unit 均为
AUDIO+listen+空文本，三条 AUDIO 路径的所有实际生成 unit 均为 AUDIO。
模型源文件 SHA256 仍为 25359f891d1aa81de16b51a24a97b8c8262b83ce3920c36b92be441fde8e4282。
GPU 0 已释放。
