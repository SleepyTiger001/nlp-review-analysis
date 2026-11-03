"""Embedding 模型的下载与文本编码。

模型权重统一放在 models/<模型名>/，不写 Hugging Face 全局缓存：
config 里写仓库名时先看本地有没有，没有才下载；写本地目录则直接用。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from config import Config
from data import asset_slug


def resolve_embedding_model(cfg: Config) -> Path:
    """返回本地模型目录，必要时下载到 models/ 下。"""
    spec = cfg.embedding.model
    local = Path(spec)
    if local.exists():
        return local.resolve()

    target = cfg.model_dir / asset_slug(spec)
    if not (target / "config.json").exists():
        print(f"下载 Embedding 模型 {spec} → {target}")
        try:
            snapshot_download(repo_id=spec, local_dir=str(target))
        except Exception as exc:  # 网络问题给出可操作的提示
            raise RuntimeError(
                f"Embedding 模型下载失败：{exc}\n"
                "先设置镜像再重试：export HF_ENDPOINT=https://hf-mirror.com，"
                f"或手动把模型文件放到 {target}"
            ) from exc
    return target


def load_encoder(cfg: Config, device: str) -> SentenceTransformer:
    path = resolve_embedding_model(cfg)
    print(f"加载 Embedding 模型：{path}（device={device}）")
    return SentenceTransformer(str(path), device=device)


def encode_texts(
    encoder: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    title: str = "",
    chunk_chars: int | None = None,
) -> np.ndarray:
    """编码为 L2 归一化向量；归一化后点积即余弦相似度。

    chunk_chars 为 None 时整篇编码，超出模型上限的部分会被截断；
    设置后按字符切块、分别编码再取平均，适合长评论。
    """
    if title:
        print(f"编码{title}：{len(texts)} 条")

    if not chunk_chars:
        vectors = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    pieces: list[str] = []
    owners: list[int] = []
    for index, text in enumerate(texts):
        chunks = [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]
        pieces.extend(chunks or [text])
        owners.extend([index] * len(chunks or [text]))
    print(
        f"  按 {chunk_chars} 字切块：{len(pieces)} 块，"
        f"平均 {len(pieces) / len(texts):.1f} 块/条"
    )

    encoded = encoder.encode(
        pieces,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    encoded = np.asarray(encoded, dtype=np.float32)

    vectors = np.zeros((len(texts), encoded.shape[1]), dtype=np.float32)
    counts = np.bincount(np.asarray(owners), minlength=len(texts))
    for index, vector in zip(owners, encoded, strict=True):
        vectors[index] += vector
    vectors /= np.maximum(counts, 1)[:, None]
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors
