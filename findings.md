官方原版已成功加载，约24GB显存；text_list签名存在。
纯文本600与437均命中，但首unit出现think特殊片段；已有静音音频unit后注入437能输出金额。
无资料时编造400元，不能视作具备可靠拒答。
评价辅助函数2个测试通过；初始红灯为evaluation模块缺失，随后实现。

初轮探索结果保留outputs/official。发现duplex.prepare未清底层model.audio_past_key_values；第二轮用model.reset_session公开API隔离用例。原探索warmup自然生成了短语，因此第二轮用公开force_listen_count=1建立纯监听unit。原探索ASR显示文字/音频可能不一致；待隔离重跑。BF16只对最后hidden行做lm_head会有0.125舍入差；第二轮使用完整feed hidden形状计算对照，避免数值误判。

最终：主组+补充共13场景14回答；官方接口的知识写入及数值覆盖成立。已有监听unit后独立文本注入和真人语音后独立资料unit为最有用正例；冷启动text437音频回转失败。两个无依据用例均编造400元。混合pending_logits与最后文本forward差值41左右，两个配对控制首token仍listen、最终文字相同，未证明刷新带来稳定改善。真人语音混合注入忽略细节，独立注入使用阳光/坚果/草坪三项。全量证据和边界见REPORT.md。

RAG扩展：GPU0当前空闲；现有TF-IDF与证据卡可迁移。旧ASR attention_mask构造不宜照搬，将用本地WhisperProcessor返回的mask与显式language/task。CLI支持query/audio/interactive和retrieve-only，完整连续麦克风全双工服务不属于本次CLI范围。

## 官方入口 RAG（2026-11-16）
- 原 TF-IDF 在忽略未知词后，可能仅凭“实验”给量子问题较高分；加入共享双字片段过滤。
- 同一会话真实检索热更新：600→437 通过，WAV 独立转写的金额相符。
- 真人文件 ASR 和检索通过，保留原音频上下文后官方模型空回答结束；状态 incomplete。
- 新 CLI 是串行文件/文字实验入口，不是实时麦克风服务。详见 RAG_REPORT.md。
