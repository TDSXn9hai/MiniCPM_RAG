2026-11-16：已完成官方入口独立验证。
主组10场景/11回答，补充组3场景/3回答；14份真实WAV均ASR回转。
源码保持原版，单独标记refreshed运行时对照；源码校验、原始目录diff均确认。
2个评价单测通过，34项产物检查通过；错误语义结果已单独报告，不宣称整体RAG通过。
GPU2峰值allocated约22.98GiB，运行结束显存已为0。无下载、无新增依赖、无进程终止。
报告REPORT.md；复现README.md；机器汇总outputs/summary.json。

## 2026-11-16 RAG 完成记录
新增 rag_core.py、rag_pipeline.py、official_runtime.py、run_rag.py、知识副本及 tests。
完成真实文本热更新/拒答验证、WAV 独立 ASR、22 项自动化测试。
音频上下文生成失败已在报告中保留，未作成功宣称。新增 RAG_GUIDE.md 和 RAG_REPORT.md。

AUDIO门控与prompt末尾0/1/2换行矩阵完成，10组真实模型对照及生成WAV独立转写。
仅新增实验脚本/报告，没有修改正式runtime或模型，也未上传新实验至GitHub。

同卡异步ASR/RAG改造完成：顺序启动加载、并行推理，全应用prefill强制AUDIO。
真实CLI、受控检索延迟门控、多轮金额热更新回归及WAV独立转写完成，31项测试通过。
保留启动meta tensor失败记录和自然时序RAG泛化回答限制。说明见RAG_GUIDE.md和ASYNC_AUDIO_RAG_REPORT.md。
