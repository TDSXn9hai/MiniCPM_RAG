# 副本同步记录

## 副本同步更新：标题检索与音频无证据行为

本次从 omni_bench_eval/minicpm_official_rag_test 同步四个文件的行为变化：
async_audio_rag.py、official_runtime.py、rag_pipeline.py 和音频流水线测试。

- 默认 title_only=True：仅标题建立TF-IDF索引，双字过滤也仅比较标题，正文仍作为证据返回。
- CLI新增 --full-text 显式使用标题+正文检索；API可用 RetrievalConfig(title_only=False)。
  差旅样例问题“上海出差住宿能报多少”需 --full-text，否则默认标题检索可能无命中。
- 音频路径无可用证据时仍注入空 reference，维持 AUDIO 注入门控后由模型决定回答或拒绝。
  返回 no_evidence=true、application_refusal_enabled=false、generation_called=true；
  status按正常结束及非空文字判定 answered/incomplete，不代表有证据或事实正确。
- 文字路径的应用层固定拒答和 retrieve-only 行为保持原有逻辑。
- 空参考资料下模型可能输出无依据内容；该变化是取消音频应用拒答，不是事实性改善。
- 差旅多轮集成脚本明确使用 --full-text；新增CLI测试覆盖默认标题模式与全文选项。

旧报告记录的是当时实现，本节更新覆盖其中“音频无证据不生成”的历史描述。
本次验证为单元测试及编译检查，没有重新运行真实GPU生成实验。
