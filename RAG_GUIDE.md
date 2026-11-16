# 基于官方 text_list 的 RAG

新增 `run_rag.py`，将本地检索结果与用户问题组织成文本，经官方全双工
`streaming_prefill(text_list=[...])` 输入。模型源码未修改，不调用私有 KV 注入方法。
这次交付的是可复用的检索层、模型适配层和串行 CLI；没有接入实时麦克风、打断或数字人服务。

## 快速使用

在本独立仓库根目录执行，先按 README 安装依赖并准备模型，确认 GPU 0 空闲：

```bash
# 仅检索，不加载 MiniCPM、不占 GPU。
python run_rag.py \
  --query '上海出差住宿能报多少' --retrieve-only

# 读取原 minicpm_rag_test 的知识文件，并生成文字和语音。
CUDA_VISIBLE_DEVICES=0 python -u run_rag.py \
  --knowledge /path/to/minicpm_rag_test/knowledge/sample.json \
  --query '上海出差住宿能报多少'

# 同一模型会话连续提问，每轮重新检查知识文件内容。
CUDA_VISIBLE_DEVICES=0 python -u run_rag.py \
  --knowledge /path/to/minicpm_rag_test/knowledge/sample.json --interactive
```

交互命令 `/reset` 在下一次模型调用前清理会话，`/quit` 退出。默认知识文件是
本项目的 `knowledge/sample.json` 独立副本；编辑原项目文件时请明确传入 `--knowledge`。
也可以重复传入 `--query`。默认输出为 `outputs/rag/<UTC时间>/`；`--output` 可指定
新目录，已有 results.json 的目录会被拒绝覆盖。

## 数据流与迁移

```text
文字问题 / 音频文件 → 本地 Whisper 转写（仅音频）
→ 知识文件内容校验与 SHA256 检查 → 字符 n-gram TF-IDF 检索
→ 分数及共享双字片段过滤 → 有长度上限的证据卡
→ 官方独立 text_list unit → streaming_generate → 分块文字和 24kHz WAV
```

从 `minicpm_rag_test/rag_core.py` 迁移了文档结构、字符 2–4 gram TF-IDF、余弦相似度、
证据卡和引用组织方式。没有迁移旧项目的私有 KV 注入、服务线程和强制说话逻辑。
新增引用中的原始 document_id、输入校验、知识文件热更新与逐轮可追溯记录。

默认参数：`--top-k 3 --min-score 0.08 --min-shared-bigrams 2 --max-evidence-chars 900`。
相似度不是置信概率。未知词会被 TF-IDF 忽略，因此另加共享双字片段过滤，减少仅因
“实验”等泛词命中无关资料的情况；两字问题只需一个片段。这是词面检索，仍会有误召回、
漏召回和截断证据风险，没有语义重排或多轮问题改写。较大知识库应先离线评估召回质量。
知识 JSON 含非空 documents 数组，每项包含唯一 id、title、content 和可选 source。
文件内容变化后重建索引；无效更新直接报错，不悄悄使用旧数据。

无可用证据时应用层固定返回“当前知识库没有足够依据回答这个问题，请补充相关资料。”，
不调用模型，不生成拒答 WAV。有证据时提示模型据资料回答；引用表示送入的证据，
不代表已经逐句验证生成内容。

模型按 `reset_session → prepare → 静音监听 unit` 建立新会话，同一会话后续问题保留上下文。
资料和问题一起放在单独的 text_list unit 中，避免此前发现的音频与文本混合同次 prefill 问题。
后续生成 unit 输入一秒静音。模型适配器必须由单一调用者串行使用；生成中禁止再次调用或重置。
异常、流被关闭或生成达到 `--max-units` 上限时，下一次调用会重建会话。

## 接口与输出

`RAGRetriever.retrieve(query)` 返回命中、证据、来源、文件 SHA256 与检索耗时。
`RAGPipeline.answer(query, on_chunk=...)` 执行检索和生成；也可以先 retrieve，再调用
`answer_retrieved(...)`。`OfficialDuplexRuntime.stream_answer(...)` 产出文字、音频和 unit 指标。
入口可参考 `run_rag.py`，调用方负责在结束时 close runtime。

输出包含 results.json、turn_*.json、events.jsonl 和有实际音频时的 turn_*.wav。
状态区分 answered、no_evidence、incomplete、error。正常完成与无证据退出码 0，
生成不完整退出码 2，异常退出码 1。文字和音频分块输出；引用及文件 hash 留在 JSON 中。

## 音频文件的当前限制

支持 `--audio <wav> --asr-device cpu`，输入最多 30 秒，使用本地 Whisper-small（默认 models/whisper-small，或 --asr-model 指定）。
先完整转写和检索，再以监听方式输入原始音频，最后单独输入证据与问题。
这条 CLI 路径不提供实时全双工调度。

本次真实音频测试中，转写正确、检索正确，但模型在注入后返回空文本并结束，
结果为 incomplete，未生成 WAV。因此当前验证通过的是文本问题驱动的官方全双工 RAG；
音频上下文加文本证据的回答稳定性仍需单独排查，不能据此宣称音频端到端已可用。
验证记录见 [RAG_REPORT.md](RAG_REPORT.md)。
