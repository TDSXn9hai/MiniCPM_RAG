# RAG 实现验证记录（旧版串行入口）

当前异步ASR/RAG、全程AUDIO实现见 ASYNC_AUDIO_RAG_REPORT.md 与 RAG_GUIDE.md。

日期：2026-11-16。使用已有本地 MiniCPM 权重、Token2Wav、Whisper-small 和根目录
`.venv`，未下载模型、未安装依赖。真实模型实验使用 GPU 0。

## 实际结果

| 场景 | 结果 | 证据 |
|---|---|---|
| 上海住宿问题，真实检索与生成 | 回答“上海出差住宿可以报销每晚六百元。” | outputs/rag_integration/turn_001.json |
| 同一会话修改知识文件后再次提问 | 回答“上海出差住宿可以报销每晚四百三十七元。”，文件 hash 改变 | outputs/rag_integration/turn_002.json |
| 量子纠缠问题 | no_evidence，应用层固定拒答，不调用模型、不生成 WAV | outputs/rag_integration/turn_003.json |
| 独立 Whisper 回转第一份回答 WAV | “嗯,上海出殺住宿可以爆箱每碗600元” | outputs/rag_integration/asr_results.json |
| 独立 Whisper 回转第二份回答 WAV | “2.4 上海出差住宿可以包銷每晚437元” | 同上 |
| 真人音频“当出现植物大战僵尸的时候提醒我” | 转写正确，命中 plant-zombies，分数 0.47286；模型首个回答 unit 为空并结束，incomplete，退出码 2 | outputs/rag_audio/results.json |

437 元仅为集成测试构造的资料更新，不代表真实政策。测试只修改了
outputs/rag_integration/knowledge.json 私有副本。语音回转金额一致，但存在明显同音字
和开头识别误差；没有据此宣称语音质量完全通过。音频问题端到端生成未通过，需继续排查
模型音频上下文与文本证据的交互。当前可用范围是文本问题驱动的检索和官方全双工生成。

## 自动化验证

```bash
cd /root/paddlejob/workspace/env_run/avatar_full/minicpm_official_rag_test
../.venv/bin/python -m pytest tests test_evaluation.py -q
```

22 项测试通过，覆盖检索、弱相关过滤、引用/证据预算、热更新、无证据不调用模型、
官方 API 调用顺序、单独文本 unit、同会话保留、异常关闭生成器、截断后会话重建和 CLI。
运行消费者写文件失败的新增测试曾发现生成器未及时关闭，现已用 finally 显式 close 修正。
Fake API 测试不能替代上述真实模型实验。

真实多轮驱动为 tests/integration_rag.py，使用同一个 CLI 子进程完成三轮断言。
它固定使用 outputs/rag_integration，已有结果时拒绝重跑，以保留此次证据。

未修改 minicpm_demo 下的源文件。官方 modeling_minicpmo.py SHA256 保持为
`25359f891d1aa81de16b51a24a97b8c8262b83ce3920c36b92be441fde8e4282`。
没有私有 decoder/KV 注入或 pending_logits 修补；原始入口对照脚本保留。

这不是实时性能基准；没有验证麦克风、用户打断、检索与收音并行或服务接入。
