# 最后一段 AUDIO 与 RAG 文本同次注入实验

2026-11-16，GPU 0，未修改模型源码或正式 RAG 入口。脚本 `test_audio_tail_rag.py`。
使用原真人音频（59629 个 16kHz 样本，约 3.727 秒），本地 Whisper-small 实际转写
“当出现植物大战僵尸的时候提醒我”，真实检索命中 plant-zombies（0.47286035）。
各组使用相同检索结果和 RAG prompt、seed 42、greedy、系统提示词及参考音色。
每组先 reset_session/prepare，并建立一秒静音监听 unit，防止跨组音频 encoder KV 泄漏。
音频分成四段，每段最多一秒，最后一段补零。前三段强制监听；所有回答阶段均未强制说话。

## 结果

| 组 | 最终输入 | 首步行为 | 完整生成文字 |
|---|---|---|---|
| separate_text | 四段音频均监听完，再独立文本 unit | turn_eos，立即结束 | 空 |
| tail_audio_only | 前三段监听，最后一段仅音频，不注入 RAG | listen；后续静音 unit 开始说话 | 好的，没问题。 |
| tail_mixed | 前三段监听，最后一段同时 audio_waveform + text_list | AUDIO 模式，先 listen；后续静音 unit 开始说话 | 好的，当出现植物大战僵尸的时候我会提醒你。 |
| tail_mixed_refreshed | 同上，仅诊断性刷新 pending_logits 为文本处理后的值 | AUDIO 模式，turn_eos，立即结束 | 空 |
| separate_text_after_tail_listen | 重复 separate_text，检查结果可复现 | turn_eos，立即结束 | 空 |
| text_without_audio | 不输入问题音频，直接输入相同问题及检索文本 | chunk_eos；后续静音 unit 开始说话 | 当用户提到植物大战僵尸时，提醒用户检查阳光、坚果冷却时间和草坪状态。 |

所有组均等待到模型自然 end_of_turn 或最多 24 units，没有忽略终止 token 强行续说。
空文本结束组仍返回了很短的音频样本；不将这些样本当作有效回答。

## 源码与 logits 观测

官方 `model/modeling_minicpmo.py`：
- 2853 附近通过输入是否包含音频选择 AUDIO；有音频优先于仅文本。
- 3100 附近，先处理音频并保存音频末尾 pending_logits。
- 3115 起，后处理 text_list；仅 TEXT 模式才从文本 feed 获取并替换 pending_logits。
  AUDIO 混合分支只写入文本对应的 KV，不更新 logits。
- 3194 起，streaming_generate 使用 pending_logits 决定首个 token。
- current_mode 在生成阶段用于 unit 的输入类型登记；没有发现“从 AUDIO 切到 TEXT 就终止”的分支。

混合组中，音频 pending_logits 的 argmax 为 `<|listen|>`，
实际最后一次文本 feed 的 argmax 为 `<|turn_eos|>`，两者最大绝对 logit 差为 43.875。
原版混合组从前者开始，刷新诊断组从后者开始。刷新只是诊断干预，不能算官方未修改行为。
独立文本组的 pending_logits 与最后文本 feed 一致，其 argmax 同样为 turn_eos。

## 结论及限制

按用户提出的方式，将最后一段音频与证据文本同次 prefill，确实使本例避免立即空回答，
但没有生成检索条目中的特有事实，尚不能称为有效 RAG 回答。
同样保持 AUDIO 模式，刷新为文本末尾 logits 后依然终止，说明 AUDIO 模式本身不足以阻止终止。
本例更直接的机制是：音频上下文加文本 prompt 后，模型下一 token 偏向 turn_eos；
原版混合路径首步使用文本之前的音频 logits，绕开了这个首步决定。
为什么该上下文下模型偏好终止，当前证据不能进一步定论；不把它认定为已完全定位的训练问题。

这仍是单条音频、单一提示词和一次检索的对照。没有验证其他问法、音频或实时异步检索。
检索在实验开始时完成，最后一段注入的时间是受控安排，不代表实时检索能赶在最后一段返回。

原始逐 unit token、模式、feed/KV 长度与 logits 比较见 outputs/audio_tail_rag/results.json，
音频独立转写见同目录 asr_results.json。输出未上传到 GitHub。

独立 Whisper 回转中，tail_mixed WAV 转写为“好的 当出现植物大战僵尸的时候我会提醒你”，
与生成文字一致；纯文本组也识别到阳光及草坪状态，但“坚果冷却”出现同音误差。
空文本组各有 2880 个 24kHz 样本（0.12 秒），Whisper 均转写为“謝謝”；
这不构成模型输出有效回答的证据，不能仅根据短音频 ASR 判断成功或确定其听感。
