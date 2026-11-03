# 文本向量 + 浅层模型流水线（分类 / 回归）

用 sentence-transformers 提取句向量，接一个浅层模型完成文本分类或数值回归。
数据集、Embedding 模型、任务类型全部由 `config.yaml` 决定，仓库本身不绑定某个模型、数据集或任务。

适合：评论情感/主题分类、评分与满意度回归、短文本打分的课程作业与基线实验。

## 能做什么

- **数据集**：Hugging Face 数据集仓库名，或本地 `csv / tsv / parquet / jsonl`
- **Embedding**：任意 sentence-transformers 模型（仓库名或本地目录），支持长文本切块平均
- **任务**：分类（`linear_svc` / `logistic_regression`）、回归（`ridge` / `linear_svr`）
- **划分**：目录内 `train/validation/test` 文件、按列切分、或自动 8:1:1 切分
- **调参**：只在 validation 上选超参数，test 只做最终报告
- **设备**：`auto` 依次尝试 `cuda → mps → cpu`，也可显式指定
- **预测**：单条、批量文件、交互式、本地网页（分类任务）

## 1. 环境准备

使用 conda，Python 3.11。下面以环境名 `torch-env` 为例：

```bash
conda create -n torch-env python=3.11 -y
conda activate torch-env

# 先按显卡安装对应的 torch（示例为 CUDA 12.4；纯 CPU 机器跳过 --index-url）
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. 三分钟跑通

仓库自带一个分类示例配置（ChnSentiCorp 酒店评论情感）：

```bash
python main.py prepare     # 下载数据集与 embedding 模型到项目目录
python main.py train       # 编码、调参、训练、测试集评估
python main.py predict --text "这家酒店位置很好，服务也很周到。"
```

训练结束会打印 Accuracy、Macro-F1、分类报告和混淆矩阵，产物写入 `artifacts/`。
参考结果：ChnSentiCorp 测试集 Accuracy 0.8867、Macro-F1 0.8866（`bge-small-zh-v1.5` + `LinearSVC`，C=1.0）。

设备默认 `auto`：有 CUDA 用 CUDA，其次 Apple MPS，最后 CPU。也可以显式指定：

```bash
python main.py train --device cuda
python main.py train --device cpu
```

## 3. 数据与模型放在哪里

**两者都落在仓库目录内，不写 `~/.cache/huggingface`，也不依赖全局缓存。**

```
data/<数据集名>/...     # Hugging Face 数据集快照，如 data/ChnSentiCorp/
models/<模型名>/...     # Embedding 权重，如 models/bge-small-zh-v1.5/
```

`python main.py prepare` 会自动下载到上述位置。也可以手动下载：

```bash
huggingface-cli download --repo-type dataset lansinuote/ChnSentiCorp \
  --local-dir data/ChnSentiCorp
huggingface-cli download BAAI/bge-small-zh-v1.5 \
  --local-dir models/bge-small-zh-v1.5
```

国内网络可先设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

`data/`、`models/`、`artifacts/` 都已写进 `.gitignore`：数据集和权重不会进入仓库，换机器时重新下载即可。

## 4. 换成自己的数据集

### 4.1 数据格式

支持 `csv / tsv / parquet / jsonl`，一行一条样本，至少包含一列文本和一列标签。

`jsonl` 示例（推荐，长文本不用处理引号和换行转义）：

```json
{"text": "故事松散，表演用力过猛。", "score": 2}
{"text": "节奏很好，结尾收得漂亮。", "score": 5}
```

### 4.2 分类任务

```yaml
task: classification
data:
  source: local
  path: data/my_reviews.jsonl
  text_column: text
  label_column: label
  label_names: ["负向", "正向"]   # 可选，顺序对应标签 0、1
```

标签可以是整数或字符串（如 `positive` / `negative`），程序会统一编码成 0..N-1。

### 4.3 回归任务

复制 `configs/regression.yaml` 修改即可：

```yaml
task: regression
data:
  source: local
  path: data/ratings.jsonl
  text_column: text
  label_column: score
model:
  name: ridge
  param_grid:
    alpha: [0.1, 1.0, 10.0, 100.0]
output:
  dir: artifacts/regression
```

```bash
python main.py train -c configs/regression.yaml
```

回归会报告 R²、RMSE、MAE、Pearson、Spearman。

### 4.4 三种划分方式

按顺序匹配，满足一种即可：

1. 目录下存在 `train.csv`、`validation.csv`、`test.csv`（或 `train-00000-of-00001.parquet` 这类文件名）
2. 单个文件里有一列划分标记，在配置里写 `split_column: split`（取值支持 `train / validation / dev / val / test`）
3. 都没有时自动按 8:1:1 随机切分（分类任务尽量分层）

### 4.5 长文本

Embedding 模型有最大输入长度，长评论会被截断。设置切块后按块编码再平均：

```yaml
embedding:
  chunk_chars: 380   # null 表示整篇编码
```

## 5. 换成自己的 Embedding 模型

改一行即可，维度自动读取：

```yaml
embedding:
  model: BAAI/bge-large-zh-v1.5   # 仓库名，自动下载到 models/bge-large-zh-v1.5
  # model: models/my-finetuned-model   # 或本地目录
```

离线场景先把权重放到 `models/`，再把 `model` 写成该目录。

## 6. 配置项

| 字段 | 说明 |
| --- | --- |
| `task` | `classification` 或 `regression` |
| `data.source` | `huggingface` 或 `local` |
| `data.name` | HF 数据集仓库名，如 `lansinuote/ChnSentiCorp` |
| `data.path` | 本地文件或目录（相对仓库根目录） |
| `data.text_column` / `label_column` | 文本列、标签列，大小写不敏感 |
| `data.split_column` | 单文件切分依据的列 |
| `data.splits` | 逻辑划分 → 文件名前缀，默认 `train/validation/test` |
| `data.max_samples` | 每个划分最多取多少条，调试用 |
| `data.label_names` | 分类类别显示名，顺序对应 0..N-1 |
| `embedding.model` | 模型仓库名或本地目录 |
| `embedding.batch_size` | 编码批大小，显存不足时调小 |
| `embedding.chunk_chars` | 长文本切块字数，`null` 为整篇 |
| `model.name` | `linear_svc` / `logistic_regression` / `ridge` / `linear_svr` |
| `model.param_grid` | 调参网格，留空用内置默认值 |
| `model.scoring` | 调参指标，留空时分类用 `f1_macro`、回归用 `neg_root_mean_squared_error` |
| `runtime.device` | `auto` / `cuda` / `mps` / `cpu` |
| `runtime.seed` | 划分与模型随机种子 |
| `output.dir` | 产物目录，默认 `artifacts` |

## 7. 命令

```bash
python main.py prepare                  # 只下载数据与模型
python main.py train                    # 训练 + 评估
python main.py train --limit 300        # 每个划分只取 300 条，快速验证环境
python main.py train --recompute        # 忽略向量缓存重新编码
python main.py predict --text "..."     # 单条预测
python main.py predict --file data/x.jsonl --out preds.csv
python main.py chat                     # 交互预测
python main.py embed "..."              # 查看句向量
python main.py train -c configs/regression.yaml   # 指定配置
```

## 8. 产物

| 文件 | 内容 |
| --- | --- |
| `artifacts/model.joblib` | 分类/回归模型、embedding 模型名、维度、最佳参数、标签名、指标 |
| `artifacts/embeddings.npz` | 三个划分的向量与标签（带配置签名，配置变了自动重算） |
| `artifacts/metrics.json` | 测试集指标、数据来源、划分规模、最佳参数 |
| `artifacts/test_predictions.csv` | 测试集逐条预测（原文、真实值、预测值、decision function） |
| `artifacts/config.yaml` | 训练时使用的配置副本，便于复现 |
| `artifacts/space_tsne.npz` | 网页工作台的 t-SNE 投影缓存 |

## 9. 目录结构

```
.
├── main.py              # 入口与命令实现：prepare / train / predict / chat / embed
├── config.py            # 配置解析与设备选择
├── data.py              # 数据集下载、读取、划分、标签编码
├── encoder.py           # embedding 模型下载与编码（含长文本切块）
├── modeling.py          # 模型注册、验证集调参、指标
├── webapp.py            # 本地网页工作台（可选，分类任务）
├── config.yaml          # 默认配置（ChnSentiCorp 分类示例）
├── configs/
│   └── regression.yaml  # 回归任务模板
├── templates/index.html
├── data/                # 数据（不入库）
├── models/              # embedding 权重（不入库）
└── artifacts/           # 训练产物（不入库）
```

## 10. 网页工作台（可选）

```bash
python main.py train       # 先生成 artifacts/model.joblib
python webapp.py           # http://127.0.0.1:7860
```

包含：交互预测、decision function 标尺、句向量热力图、测试集 t-SNE 散点（可点选）、最近邻样本、决策值分布。
只监听 `127.0.0.1`，数据不出本机。工作台面向二分类任务；回归任务请用 `predict` 命令。

## 11. 常见问题

1. **`指定了 cuda，但 PyTorch 检测不到可用 GPU`**：装的 torch 是 CPU 版，按官网命令重装对应 CUDA 版本；或改用 `--device cpu`。
2. **显存不足**：调小 `embedding.batch_size`；长文本先调小 `embedding.chunk_chars`。
3. **下载慢或超时**：设置 `HF_ENDPOINT=https://hf-mirror.com`，或用 `huggingface-cli` 手动下载到 `data/`、`models/`。
4. **指标和换模型前对不上**：向量缓存按配置签名复用，改模型或数据后会自动重算；若权重内容变了但名字没变，加 `--recompute`。
5. **标签是字符串**：无需改代码，会自动按字典序编码；用 `data.label_names` 指定可读的类别名。
6. **macOS MPS 报错**：用 `--device cpu` 绕过，或升级 torch 与系统版本。

## 12. 开发

```bash
ruff format . && ruff check .
mypy main.py config.py data.py encoder.py modeling.py
```

## 13. 数据与模型来源

- 示例数据集：[`lansinuote/ChnSentiCorp`](https://huggingface.co/datasets/lansinuote/ChnSentiCorp)（中文酒店评论情感）
- 示例模型：[`BAAI/bge-small-zh-v1.5`](https://huggingface.co/BAAI/bge-small-zh-v1.5)

仓库不附带数据与权重。公开分发数据或用于商业场景前，请自行核对各自的使用条款。
