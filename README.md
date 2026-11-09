# 文本向量 + 浅层模型流水线（分类 / 回归）

用 sentence-transformers 提取句向量，接浅层模型完成文本分类（LinearSVC / LogisticRegression）或数值回归（Ridge / LinearSVR）。
数据集、Embedding 模型、任务类型全部由 `configs/*.yaml` 决定，仓库本身不绑定某个模型、数据集或任务。

## 1. 环境准备

使用 conda，Python 3.11。下面以环境名 `torch-env` 为例：

```bash
conda create -n torch-env python=3.11 -y
conda activate torch-env

# 先按显卡安装对应的 torch（示例为 CUDA 12.4；纯 CPU 机器跳过 --index-url）
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## 2. 跑通分类（默认配置）

```bash
python main.py prepare                                # 下载数据集与 embedding 模型到项目内
python main.py train -c configs/classification.yaml   # 编码、调参、训练、测试集评估
python main.py predict -c configs/classification.yaml --text "这家酒店位置很好，服务也很周到。"
python webapp.py -c configs/classification.yaml       # 网页工作台，http://127.0.0.1:7860
```

训练打印 Accuracy、Macro-F1、分类报告和混淆矩阵。参考结果：ChnSentiCorp 测试集 Accuracy 0.8867、
Macro-F1 0.8866（`bge-small-zh-v1.5` + `LinearSVC`，C=1.0）。
设备默认 `auto`（cuda → mps → cpu），可用 `--device` 显式指定。

## 3. 跑通回归

豆瓣长评 → 评分（1-5）的示例配置。数据不随仓库分发，先放到 `data/niulai/clean.jsonl`：

```bash
python main.py train -c configs/regression.yaml   # 报告 R²、RMSE、MAE、Pearson、Spearman
python webapp.py -c configs/regression.yaml
```

## 4. 数据与模型放在哪里

两者都落在仓库目录内，不写 `~/.cache/huggingface`：

```
data/<数据集名>/...     # Hugging Face 数据集快照
models/<模型名>/...     # Embedding 权重
```

`prepare` 自动下载（仓库同时提供 bin 与 safetensors 权重时只下后者），也可以手动：

```bash
hf download --repo-type dataset lansinuote/ChnSentiCorp --local-dir data/ChnSentiCorp
hf download BAAI/bge-small-zh-v1.5 --local-dir models/bge-small-zh-v1.5
```

国内网络可先 `export HF_ENDPOINT=https://hf-mirror.com`。`data/`、`models/`、`artifacts/` 均在 `.gitignore` 中，换机器重新下载即可。

## 5. 换成自己的数据与模型

本地数据支持 `csv / tsv / parquet / jsonl`（每行一条）与 `json`（记录数组），至少一列文本、一列标签，列名大小写不敏感：

```yaml
# configs/my.yaml —— 分类；回归把 task 改为 regression，label_column 换成数值列
task: classification
data:
  source: local
  path: data/my_reviews.jsonl
  text_column: text
  label_column: label
  label_names: ["负向", "正向"]   # 可选，顺序对应标签 0、1
```

三种划分方式按优先级，满足其一即可：

1. 目录内有 `train / validation / test` 文件（`train-00000-of-00001.parquet` 这类前缀也认，前缀映射可用 `data.splits` 改）
2. 单文件里有划分列时写 `data.split_column`（取值 train / training / validation / valid / val / dev / test，三类需各出现至少一次）
3. 都没有时自动按 8:1:1 随机切分，分类任务尽量分层

长评论设置 `embedding.chunk_chars: 380` 按块编码再平均（`null` 表示整篇编码，超长截断）。
换 Embedding 模型改 `embedding.model` 一行：仓库名自动下载到 `models/`，或直接指向本地目录，维度自动读取。

## 6. 配置项速查

| 字段 | 说明 |
| --- | --- |
| `task` | `classification` 或 `regression` |
| `data.source` | `huggingface` 或 `local` |
| `data.name` / `data.path` | HF 数据集仓库名 / 本地文件或目录（相对仓库根目录） |
| `data.text_column` / `data.label_column` | 文本列、标签列，大小写不敏感 |
| `data.split_column` | 单文件按列切分；train/validation/test 三类值需各出现至少一次 |
| `data.splits` | 逻辑划分 → 文件名前缀，默认 `train/validation/test` |
| `data.max_samples` | 每个划分最多取前多少条，调试用 |
| `data.label_names` | 分类别名显示名，顺序对应 0..N-1；回归忽略 |
| `embedding.model` | 模型仓库名或本地目录 |
| `embedding.batch_size` | 编码批大小，显存不足时调小 |
| `embedding.chunk_chars` | 长文本切块字数，`null` 为整篇 |
| `model.name` | 分类 `linear_svc` / `logistic_regression`；回归 `ridge` / `linear_svr` |
| `model.param_grid` | 调参网格，留空用内置默认值 |
| `model.scoring` | 调参指标，留空分类用 `f1_macro`、回归用 `neg_root_mean_squared_error` |
| `model.refit_full` | `true` 时评估后用全量数据重训交付模型（指标仍来自留出 test） |
| `runtime.device` | `auto` / `cuda` / `mps` / `cpu` |
| `runtime.seed` | 划分与模型随机种子 |
| `output.dir` | 产物目录，默认 `artifacts` |

## 7. 命令

```bash
python main.py prepare                                  # 只下载数据与模型
python main.py train -c configs/classification.yaml     # 训练 + 评估
python main.py train -c configs/classification.yaml --limit 300   # 每个划分取前 300 条，快速验证
python main.py train -c configs/classification.yaml --recompute   # 忽略向量缓存重新编码
python main.py predict -c configs/classification.yaml --text "..."          # 单条预测
python main.py predict -c configs/classification.yaml --file data/x.jsonl --out preds.csv  # 批量
python main.py chat -c configs/classification.yaml      # 交互预测
python main.py embed -c configs/classification.yaml "..."          # 查看句向量
python webapp.py -c configs/classification.yaml         # 网页工作台（另有 --port / --no-browser / --device）
```

回归任务把 `-c` 换成 `configs/regression.yaml`。

## 8. 产物

| 文件 | 内容 |
| --- | --- |
| `artifacts/model.joblib` | 模型（`refit_full: true` 时为全量重训版）、embedding 模型名与维度、最佳参数、标签名、指标 |
| `artifacts/embeddings.npz` | 三个划分的向量与标签（带「embedding 配置 + 数据内容摘要」签名，配置或数据变了自动重算） |
| `artifacts/metrics.json` | 测试集指标、数据来源、划分规模、最佳参数 |
| `artifacts/test_predictions.csv` | 测试集逐条预测（原文、真实值、预测值、decision function） |
| `artifacts/config.yaml` | 训练时使用的配置副本，便于复现 |
| `artifacts/space_tsne.npz` | 网页工作台的 t-SNE 投影缓存 |

## 9. 网页工作台

交互预测、句向量热力图、测试集 t-SNE 散点（可点选）、最近邻样本、模型输出分布、测试指标，
以及「词云对比」标签页（好评 / 差评两张词云 + 特色词对比，见第 12 节）。
分类显示类别与 decision function 标尺；多于两类的多分类自动降级为按「预测正确 / 错误」着色；
回归显示预测评分，散点按真实标签高低着色。
`refit_full: true` 时，工作台的判定来自全量重训后的交付模型（对测试集属 in-sample 口径），顶部指标徽章仍来自留出测试集。
默认只监听 `127.0.0.1`（可用 `--host` 改绑）。

## 10. 常见问题
1. **cuda / MPS 报错**：cuda 提示「检测不到 GPU」说明装的是 CPU 版 torch，按官网命令重装；MPS 报错可先用 `--device cpu` 绕过。
2. **显存不足**：调小 `embedding.batch_size`；长文本再调小 `embedding.chunk_chars`。
3. **下载慢或超时**：设 `HF_ENDPOINT=https://hf-mirror.com`，或用 `hf download` 手动下载。
4. **指标和之前对不上**：向量缓存按「embedding 配置 + 数据内容摘要」复用，改模型、改数据、改种子会自动重算；若 embedding 权重内容变了但仓库名没变，加 `--recompute`。
5. **标签是字符串**：无需改代码，自动按字典序编码；用 `data.label_names` 指定显示名。

## 11. 开发与来源

```bash
ruff format . && ruff check .
mypy main.py config.py data.py encoder.py modeling.py webapp.py
```

- 示例数据集：[`lansinuote/ChnSentiCorp`](https://huggingface.co/datasets/lansinuote/ChnSentiCorp)（中文酒店评论情感）
- 示例模型：[`BAAI/bge-small-zh-v1.5`](https://huggingface.co/BAAI/bge-small-zh-v1.5)

仓库不附带数据与权重；公开分发或商用前请自行核对各自的使用条款。

## 12. 豆瓣影评回归（作业增量）

在原有流水线上增量实现了「豆瓣短评 → 星级」的完整链路，电影为《给阿嬷的情书》（豆瓣 subject 37116446）。

| 新增文件 | 作用 |
| --- | --- |
| `scripts/douban_login.py` | 用本机 Edge 打开有头窗口，由用户本人登录，检测到 `dbcl2` cookie 后把登录态存为 `data/douban/storage_state.json`，并在同一会话内验证短评页 |
| `scripts/douban_scrape.py` | 复用登录态翻页抓短评，解析文本 / 星级（`allstar*` 换算 1-5）/ 时间，翻页间隔 3.5 秒 + 随机抖动，输出 `data/douban/<电影名>.jsonl` |
| `configs/douban.yaml` | `task: regression`、`source: local`、`chunk_chars: 300`、Ridge 调参网格 |
| `wordcloud_panel.py` | jieba 分词 + 停用词 + wordcloud 出图，按 4-5 星 / 1-2 星分组，并按每万词频挑出对比词 |
| `assets/stopwords_zh.txt` | 影评场景停用词表（375 词），滤掉虚词与「电影/一部/这部」这类无区分度泛称 |

**评阅直接跑这三条**（数据集与 embedding 权重都已随仓库提供，clone 后无需联网下载、无需爬取）：

```bash
python main.py train -c configs/douban.yaml     # 训练 + 五指标，CPU 约 1 分钟
python main.py embed -c configs/douban.yaml "测试"   # 验证：512 维、L2 范数 1.000000
python webapp.py -c configs/douban.yaml         # 工作台，顶部「词云对比」标签页
```

`models/bge-small-zh-v1.5/`（约 92MB）与 `data/douban/给阿嬷的情书.jsonl`（391 条）都在仓库内。
若权重被误删，`train` 会自动重新下载；下载不畅可设镜像 `export HF_ENDPOINT=https://hf-mirror.com`。
登录态 `data/douban/storage_state.json` 含账号 cookie，**不在仓库内**，也不应提交。

**想自己重新爬取**（可选，需要你自己的豆瓣登录态，仓库不含）：

```bash
python scripts/douban_login.py                  # 有头浏览器由本人登录，存 data/douban/storage_state.json
python scripts/douban_scrape.py --target 420    # 抓短评，豆瓣该片约 400 条封顶
```

工作台新增 `GET /api/wordcloud`：返回两组词云的 PNG（base64）、高频词与对比词，结果进程内缓存（首次约 2-3 秒）。
`webapp.py` 同时修了两处：测试集小于 50 条时 `PCA(n_components=50)` 会直接报错（改成 `min(50, n-1, dim)`，perplexity 同步自适应），以及开启 `TEMPLATES_AUTO_RELOAD` 让改模板不必重启。

**本次数据与结果**：391 条短评（1★34 / 2★27 / 3★75 / 4★80 / 5★175），按 8:1:1 切分为 312/39/40。
测试集 R² 0.435、RMSE 1.019、MAE 0.804、Pearson 0.717、Spearman 0.670。
预测值明显向均值压缩（预测标准差 0.68，真实 1.36）：真实 1★ 的平均预测 2.75、5★ 的 4.28，方向对但强度不足，
单条极端短评会互相越界，适合判断倾向而非给出精确分数。登录态文件 `data/douban/storage_state.json` 含账号 cookie，
已在 `.gitignore` 覆盖范围内，不要提交或外传。
