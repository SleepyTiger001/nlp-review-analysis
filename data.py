"""数据集下载、读取与标签编码。

目录约定（全部落在仓库内，不使用 ~/.cache/huggingface）：

    data/<数据集名>/...   Hugging Face 数据集快照，由 prepare 下载
    data/<任意文件>      自备数据，config 里用 data.path 指向

支持的表格格式：csv / tsv / parquet / jsonl（一行一条记录）。
划分规则按优先级：
    1. 目录内存在 train / validation / test 文件
       （`train.csv` 或 `train-00000-of-00001.parquet` 都认）；
    2. 单文件且配置了 data.split_column，按该列取值切分；
    3. 其余情况按 8:1:1 随机切分，分类任务尽量分层。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split

from config import SPLITS, Config

TABLE_SUFFIXES: tuple[str, ...] = (".csv", ".tsv", ".parquet", ".jsonl", ".json")
SPLIT_ALIASES: dict[str, str] = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "dev": "validation",
    "test": "test",
}


@dataclass
class DatasetBundle:
    """统一的训练输入：每个划分的文本与标签。"""

    texts: dict[str, list[str]]
    labels: dict[str, np.ndarray]
    label_names: dict[int, str] | None
    source: str
    location: str


def asset_slug(spec: str) -> str:
    """把仓库名或路径压成安全的目录名。

    例如 BAAI/bge-small-zh-v1.5 → bge-small-zh-v1.5。
    """
    name = Path(spec).name or spec
    return re.sub(r"[^0-9A-Za-z._-]+", "__", name).strip("_") or "asset"


def read_table(path: Path) -> pd.DataFrame:
    """按扩展名读取表格。"""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(
        f"不支持的表格格式：{path.name}（仅支持 csv / tsv / parquet / jsonl）"
    )


def prepare_dataset(cfg: Config) -> Path:
    """把数据集落到 data/ 下并返回本地路径。"""
    if cfg.data.source == "local":
        path = Path(cfg.data.path or "")
        path = path if path.is_absolute() else cfg.root / path
        if not path.exists():
            raise FileNotFoundError(f"本地数据集不存在：{path}")
        return path

    name = cfg.data.name or ""
    target = cfg.data_dir / asset_slug(name)
    if not _table_files(target):
        print(f"下载数据集 {name} → {target}")
        try:
            snapshot_download(repo_id=name, repo_type="dataset", local_dir=str(target))
        except Exception as exc:  # 网络问题给出可操作的提示
            raise RuntimeError(
                f"数据集下载失败：{exc}\n"
                "先设置镜像再重试：export HF_ENDPOINT=https://hf-mirror.com，"
                f"或手动把数据集文件放到 {target}"
            ) from exc
    return target


def load_dataset_bundle(cfg: Config, verbose: bool = True) -> DatasetBundle:
    """读取全部划分并完成标签编码。"""
    location = prepare_dataset(cfg)
    if location.is_file():
        frames = _frames_from_file(cfg, location)
    else:
        frames = _frames_from_dir(cfg, location)

    if cfg.data.max_samples:
        frames = {k: v.head(cfg.data.max_samples) for k, v in frames.items()}

    texts, labels, label_names = _encode_labels(cfg, frames)
    if verbose:
        _print_summary(texts, labels, label_names)
    return DatasetBundle(
        texts=texts,
        labels=labels,
        label_names=label_names,
        source=cfg.data.source,
        location=str(location),
    )


def _table_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in TABLE_SUFFIXES)


def _match_split(path: Path, prefix: str) -> bool:
    return path.stem == prefix or path.stem.startswith(f"{prefix}-")


def _frames_from_dir(cfg: Config, root: Path) -> dict[str, pd.DataFrame]:
    files = _table_files(root)
    if not files:
        raise FileNotFoundError(
            f"{root} 下没有 csv/tsv/parquet 文件；"
            "请先运行 python main.py prepare，或把数据放到该目录"
        )

    found: dict[str, Path] = {}
    for split in SPLITS:
        prefix = cfg.data.splits.get(split, split)
        hits = [p for p in files if _match_split(p, prefix)]
        if hits:
            found[split] = hits[0]

    if "train" not in found:
        raise FileNotFoundError(
            f"{root} 下找不到 train 划分（文件名需以 train 开头）。"
            f"现有文件：{[p.name for p in files]}"
        )

    train_df = _prepare_frame(cfg, read_table(found["train"]))
    if "validation" in found and "test" in found:
        return {
            "train": train_df,
            "validation": _prepare_frame(cfg, read_table(found["validation"])),
            "test": _prepare_frame(cfg, read_table(found["test"])),
        }

    if "test" in found:
        print(f"未找到 validation，从 train 切出 10% 作为验证集：{root}")
        train_df, val_df = train_test_split(
            train_df, test_size=0.1, random_state=cfg.runtime.seed
        )
        return {
            "train": train_df,
            "validation": val_df,
            "test": _prepare_frame(cfg, read_table(found["test"])),
        }

    if "validation" in found:
        print(f"未找到 test，从 train 切出 10% 作为测试集：{root}")
        train_df, test_df = train_test_split(
            train_df, test_size=0.1, random_state=cfg.runtime.seed
        )
        return {
            "train": train_df,
            "validation": _prepare_frame(cfg, read_table(found["validation"])),
            "test": test_df,
        }

    print(f"只找到 train，按 8:1:1 重新切分：{root}")
    return _random_split(cfg, train_df)


def _frames_from_file(cfg: Config, path: Path) -> dict[str, pd.DataFrame]:
    df = read_table(path)
    if cfg.data.split_column:
        column = match_column(df, cfg.data.split_column, "划分")
        values = df[column].astype(str).str.strip().str.lower().map(SPLIT_ALIASES)
        unknown = sorted(df.loc[values.isna(), column].astype(str).unique())
        if unknown:
            raise ValueError(
                f"划分列 {column} 出现无法识别的取值 {unknown}；"
                f"可用取值：{sorted(SPLIT_ALIASES)}"
            )
        frames = {split: df[values == split] for split in SPLITS}
        empty = [k for k, v in frames.items() if v.empty]
        if empty:
            raise ValueError(f"划分列 {column} 切分后这些划分为空：{empty}")
        return {k: _prepare_frame(cfg, v) for k, v in frames.items()}

    print(f"单文件且未配置 split_column，按 8:1:1 随机切分：{path}")
    return _random_split(cfg, _prepare_frame(cfg, df))


def _random_split(cfg: Config, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    stratify = None
    if cfg.task == "classification":
        counts = df["label"].value_counts()
        if len(counts) > 1 and int(counts.min()) >= 3:
            stratify = df["label"].astype(str)

    train_df, rest = train_test_split(
        df, test_size=0.2, random_state=cfg.runtime.seed, stratify=stratify
    )
    stratify_rest = None
    if stratify is not None:
        rest_counts = rest["label"].astype(str).value_counts()
        if int(rest_counts.min()) >= 2:
            stratify_rest = rest["label"].astype(str)
    val_df, test_df = train_test_split(
        rest, test_size=0.5, random_state=cfg.runtime.seed, stratify=stratify_rest
    )
    return {"train": train_df, "validation": val_df, "test": test_df}


def _prepare_frame(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """取出文本与标签两列，清掉空文本；回归任务把标签转成数值。"""
    text_col = match_column(df, cfg.data.text_column, "文本")
    label_col = match_column(df, cfg.data.label_column, "标签")

    out = pd.DataFrame(
        {
            "text": df[text_col].fillna("").astype(str).str.strip(),
            "label": df[label_col],
        }
    )
    if cfg.task == "regression":
        out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["label"])
    out = out[out["text"] != ""]
    if out.empty:
        raise ValueError(
            f"清洗后没有可用样本，请检查列名 {text_col}/{label_col} 是否正确"
        )
    return out.reset_index(drop=True)


def match_column(df: pd.DataFrame, name: str, kind: str) -> str:
    """列名大小写不敏感匹配，缺失时报出可用列。"""
    if name in df.columns:
        return name
    lowered = {str(c).lower(): c for c in df.columns}
    if name.lower() in lowered:
        return lowered[name.lower()]
    raise KeyError(f"找不到{kind}列 {name!r}；当前列：{[str(c) for c in df.columns]}")


def _encode_labels(
    cfg: Config, frames: dict[str, pd.DataFrame]
) -> tuple[dict[str, list[str]], dict[str, np.ndarray], dict[int, str] | None]:
    texts = {k: v["text"].tolist() for k, v in frames.items()}

    if cfg.task == "regression":
        labels = {k: v["label"].to_numpy(dtype=np.float32) for k, v in frames.items()}
        return texts, labels, None

    values: list[Any] = []
    for frame in frames.values():
        for value in frame["label"].tolist():
            if value not in values:
                values.append(value)
    if len(values) < 2:
        raise ValueError(f"分类任务至少需要 2 个类别，当前只有：{values}")
    order = _label_order(values)
    mapping = {value: index for index, value in enumerate(order)}

    labels = {
        k: np.asarray([mapping[v] for v in v["label"].tolist()], dtype=np.int64)
        for k, v in frames.items()
    }
    names = cfg.data.label_names
    if names is not None:
        if len(names) != len(order):
            raise ValueError(
                f"data.label_names 有 {len(names)} 个名字，"
                f"但数据里有 {len(order)} 个类别：{order}"
            )
        label_names = {i: str(name) for i, name in enumerate(names)}
    else:
        label_names = {i: str(value) for i, value in enumerate(order)}
    return texts, labels, label_names


def _label_order(values: list[Any]) -> list[Any]:
    """数值标签按数值排序，其余按字符串排序，保证 encode 结果稳定。"""
    try:
        return sorted(values, key=float)
    except (TypeError, ValueError):
        return sorted(values, key=str)


def _print_summary(
    texts: dict[str, list[str]],
    labels: dict[str, np.ndarray],
    label_names: dict[int, str] | None,
) -> None:
    for split in SPLITS:
        if split not in texts:
            continue
        y = labels[split]
        if label_names is None:
            stats = (
                f"标签均值 {float(np.mean(y)):.4f}｜"
                f"范围 {float(np.min(y)):.4f}~{float(np.max(y)):.4f}"
            )
        else:
            counts = np.bincount(y.astype(np.int64), minlength=len(label_names))
            detail = "，".join(
                f"{label_names[i]} {int(counts[i])}" for i in range(len(label_names))
            )
            stats = f"类别分布：{detail}"
        print(f"{split:10s}: {len(texts[split]):6d} 条｜{stats}")
