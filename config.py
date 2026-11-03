"""配置加载与计算设备选择。

配置文件是一个 YAML 文件，字段与下面的 dataclass 一一对应。
出现未知字段会直接报错，避免拼写错误被静默忽略。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

Task = Literal["classification", "regression"]

SPLITS: tuple[str, ...] = ("train", "validation", "test")
DEVICES: tuple[str, ...] = ("auto", "cuda", "mps", "cpu")
MODEL_FILE: str = "model.joblib"  # 训练产物文件名，main 与 webapp 共用


@dataclass
class DataConfig:
    """数据集来源与列名。"""

    source: str = "huggingface"  # huggingface | local
    name: str | None = None  # source=huggingface：数据集仓库名
    path: str | None = None  # source=local：csv/tsv/parquet/jsonl/json 文件或目录
    text_column: str = "text"
    label_column: str = "label"
    split_column: str | None = None  # 单文件切分列；三类划分值需各出现至少一次
    splits: dict[str, str] = field(default_factory=dict)  # 逻辑划分 -> 文件名前缀
    max_samples: int | None = None  # 每个划分最多取前多少条，调试用
    label_names: list[str] | None = None  # 分类任务的类别显示名


@dataclass
class EmbeddingConfig:
    """Embedding 模型与编码参数。"""

    model: str = "BAAI/bge-small-zh-v1.5"  # 仓库名或本地目录
    batch_size: int = 32
    chunk_chars: int | None = None  # 长文本切块字数；null 表示整篇编码（超长会被截断）


@dataclass
class ModelConfig:
    """下游浅层模型与调参网格。"""

    name: str = "linear_svc"
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    scoring: str | None = None
    refit_full: bool = False  # 评估后是否用 train+validation+test 全量重训交付模型


@dataclass
class RuntimeConfig:
    """运行环境。"""

    device: str = "auto"  # auto 依次尝试 cuda、mps、cpu
    seed: int = 42


@dataclass
class OutputConfig:
    """产物目录（相对仓库根目录）。"""

    dir: str = "artifacts"


@dataclass
class Config:
    task: str = "classification"
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    root: Path = field(default_factory=Path.cwd)

    def validate(self) -> None:
        if self.task not in ("classification", "regression"):
            raise ValueError(
                f"task 只能是 classification 或 regression，实际是 {self.task!r}"
            )
        if self.data.source not in ("huggingface", "local"):
            raise ValueError(
                f"data.source 只能是 huggingface 或 local，实际是 {self.data.source!r}"
            )
        if self.data.source == "huggingface" and not self.data.name:
            raise ValueError(
                "data.source=huggingface 时必须填写 data.name，"
                "例如 lansinuote/ChnSentiCorp"
            )
        if self.data.source == "local" and not self.data.path:
            raise ValueError("data.source=local 时必须填写 data.path")
        if self.runtime.device not in DEVICES:
            raise ValueError(f"runtime.device 只能是 {list(DEVICES)} 之一")
        unknown = sorted(set(self.data.splits) - set(SPLITS))
        if unknown:
            raise ValueError(
                f"data.splits 只能使用 {list(SPLITS)} 作为键，多出：{unknown}"
            )

    @property
    def artifact_dir(self) -> Path:
        return self.root / self.output.dir

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def model_dir(self) -> Path:
        return self.root / "models"


def _build(cls: type, raw: Any, section: str) -> Any:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"配置段 {section} 必须是键值表，实际是 {type(raw).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"配置段 {section} 存在未知字段 {unknown}；可用字段：{sorted(known)}"
        )
    return cls(**raw)


def load_config(
    path: str | Path = "configs/classification.yaml", root: Path | None = None
) -> Config:
    """读取 YAML 配置；root 为仓库根目录，默认当前工作目录。"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件 {config_path}。"
            "请在仓库根目录运行，或用 -c 指定配置文件，例如 -c configs/regression.yaml"
        )

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件 {config_path} 顶层必须是键值表")

    cfg = Config(
        task=str(raw.get("task", "classification")),
        data=_build(DataConfig, raw.get("data"), "data"),
        embedding=_build(EmbeddingConfig, raw.get("embedding"), "embedding"),
        model=_build(ModelConfig, raw.get("model"), "model"),
        runtime=_build(RuntimeConfig, raw.get("runtime"), "runtime"),
        output=_build(OutputConfig, raw.get("output"), "output"),
        root=(root or Path.cwd()).resolve(),
    )
    cfg.data.splits = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    } | cfg.data.splits
    cfg.validate()
    return cfg


def choose_device(requested: str) -> str:
    """把 auto 解析成实际设备；显式指定却不可用时给出明确报错。"""
    if requested not in DEVICES:
        raise ValueError(f"未知设备 {requested!r}，可选：{list(DEVICES)}")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "指定了 cuda，但 PyTorch 检测不到可用 GPU。"
            "请检查驱动与 torch 的 CUDA 版本，或改用 --device cpu。"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "指定了 mps，但当前 PyTorch 不支持 MPS"
            "（需要 Apple Silicon 与 macOS 12.3+）。请改用 --device cpu。"
        )
    return requested


def describe_device(device: str) -> str:
    """日志里显示的设备描述。"""
    if device == "cuda":
        return f"cuda（{torch.cuda.get_device_name(0)}）"
    if device == "mps":
        return "mps（Apple GPU）"
    return "cpu"
