目的：独立实测官方 MiniCPMODuplex.text_list；不修改原有服务/源码。
1. 环境/资产/显存核查 [完成]
2. 复制小文件、链接本地大权重；记录来源校验值 [完成：无下载]
3. 官方原版纯文本、持续音频、混合输入实测 [完成：正式主组10场景11回答]
4. pending_logits 对照、跨轮资料更新、语音回转检查 [完成：补充3场景3回答，合计14份WAV+ASR]
5. 汇总真实结果、可复现命令与限制 [完成：REPORT.md、README.md、outputs/summary.json]

验证：评价测试2 passed；14份音频、ASR和结束状态等机械检查34/34；源代码SHA一致；GPU2已释放。
语义限制：冷启动文本异常、无资料编造400、真人语音混合注入未使用细节；这些不因调用完成而视为通过。
错误/实验修正：初轮未清音频encoder KV且warmup自行发言，保留为探索组；正式组增加公开reset_session及force_listen_count。BF16最后行GEMM舍入已通过同形状计算排除。评价模块初始缺失是TDD红灯。一次系统python缺soundfile，改用项目.venv后检查完成。


## RAG扩展（2026-11-16）
目标：可复用检索+官方独立text_list unit+文本/文件语音CLI，不修改模型和旧项目。
1. [进行中] 检索、空命中拒答、官方调用顺序的失败测试。
2. [待完成] 迁移rag_core并添加知识加载/更新；实现rag_pipeline。
3. [待完成] 官方runtime适配、run_rag CLI、音频文件ASR。
4. [待完成] 单测、检索预览、GPU0真实文本/语音/拒答/437覆盖及ASR回转。
5. [待完成] 使用说明、迁移来源、真实结果和限制。
设计：知识JSON内容hash变化时重建索引；字符2-4gramTF-IDF，top_k/阈值/证据预算可配；空证据不进模型；每个模型操作同步串行；新会话reset，已有会话不每轮重置；只用text_list，不混合audio+text。
文件：rag_core.py、rag_pipeline.py、official_runtime.py、run_rag.py、knowledge/sample.json、tests/test_rag*.py；保留run_probe与历史输出。

## 2026-11-16：官方 text_list RAG
- [x] 迁移 TF-IDF、证据卡和引用结构，增加文件热更新与弱命中过滤。
- [x] 独立官方全双工运行层及 CLI；保持文本 unit 独立。
- [x] 22 项测试；回调失败关闭生成器及截断后的会话重建。
- [x] GPU 0 验证真实多轮 600→437、无依据拒答和输出 WAV 独立转写。
- [x] 真人音频验证并如实记录空回答失败；使用指南和报告完成。

## 2026-11-16：异步ASR/RAG + 全程AUDIO改造
- [x] 失败测试：禁止TEXT-only、ASR与MiniCPM重叠、注入期间listen、超时/ASR错误/空证据。
- [x] 同卡常驻Whisper后台worker与独立CUDA stream，检索在同一worker接续执行。
- [x] MiniCPM单一owner，AUDIO等待/混合注入/下一AUDIO unit释放；同步文字入口同样使用AUDIO。
- [进行中] GPU0真实CLI、重叠时间线、生成WAV及多轮回归验证。
- [待完成] 异常/取消完善、运行说明与真实限制记录。

异步改造收尾：
- [x] GPU0真实CLI重叠验证，ASR约0.5812秒/重叠0.5805秒，检索亦重叠。
- [x] 真实人为延迟检索的2-unit监听门控；AUDIO混合注入与下一unit释放，输出资料细节。
- [x] 全AUDIO文字600→437/拒答回归与生成WAV独立转写。
- [x] 31项测试及说明文档，启动并发构造meta tensor问题修复且保留原记录。
