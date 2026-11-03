"""命令行入口。

用法：
    python main.py prepare                 下载数据集与 embedding 模型到项目内
    python main.py train                   编码、调参、训练、测试集评估
    python main.py predict --text "..."    单条预测
    python main.py predict --file x.csv    批量预测
    python main.py chat                    交互预测
    python main.py embed "..."             查看句向量
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from config import (
    DEVICES,
    MODEL_FILE,
    SPLITS,
    Config,
    choose_device,
    describe_device,
    load_config,
)
from data import load_dataset_bundle, match_column, prepare_dataset, read_table
from encoder import encode_texts, load_encoder, resolve_embedding_model
from modeling import build_estimator, decision_scores, evaluate, train_model


def _config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if getattr(args, "limit", None):
        cfg.data.max_samples = args.limit
    return cfg


def _device(args: argparse.Namespace, cfg: Config) -> str:
    return choose_device(args.device or cfg.runtime.device)


def _align_embedding_model(cfg: Config, trained: dict[str, Any]) -> None:
    """预测类命令一律按训练时的 embedding 编码，否则维度或语义对不上。"""
    if cfg.embedding.model != trained["embedding_model"]:
        print(
            f"注意：配置里的 embedding 是 {cfg.embedding.model}，"
            f"训练时是 {trained['embedding_model']}，按训练时的模型加载。"
        )
        cfg.embedding.model = trained["embedding_model"]


def cmd_prepare(args: argparse.Namespace) -> None:
    cfg = _config(args)
    data_path = prepare_dataset(cfg)
    model_path = resolve_embedding_model(cfg)
    print(f"\n数据集：{data_path}")
    print(f"Embedding 模型：{model_path}")
    print("两者都在仓库目录内，拷贝或删除都只影响本项目。")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _config(args)
    device = _device(args, cfg)
    print(f"任务：{cfg.task}｜设备：{describe_device(device)}")

    bundle = load_dataset_bundle(cfg)
    encoder = load_encoder(cfg, device)
    X, y = _get_embeddings(cfg, bundle, encoder, force=args.recompute)

    result = train_model(
        task=cfg.task,
        name=cfg.model.name,
        param_grid=cfg.model.param_grid,
        scoring=cfg.model.scoring,
        seed=cfg.runtime.seed,
        X_train=X["train"],
        y_train=y["train"],
        X_val=X["validation"],
        y_val=y["validation"],
    )

    print("\n验证集调参结果：")
    for params, score in result.candidates:
        print(f"  {params} → {score:+.4f}")
    print(f"最佳参数：{result.best_params}")

    y_pred = result.estimator.predict(X["test"])
    metrics = evaluate(cfg.task, y["test"], y_pred)
    _print_eval(cfg, bundle, y["test"], y_pred, metrics, result)

    final_estimator = result.estimator
    if cfg.model.refit_full:
        final_estimator = _refit_full(cfg, result, X, y)
        print(
            "\n交付模型：按最佳参数在 train+validation+test 全量数据上重训"
            "（上面的指标仍来自留出测试集）。"
        )

    _save_artifacts(cfg, bundle, X, y, result, metrics, y_pred, final_estimator, args)


def cmd_predict(args: argparse.Namespace) -> None:
    if not args.text and not args.file:
        raise SystemExit("请提供 --text 单条预测，或 --file 批量预测")

    cfg = _config(args)
    device = _device(args, cfg)
    model_path = cfg.artifact_dir / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(
            f"没有找到 {model_path}，请先运行：python main.py train"
        )

    trained = joblib.load(model_path)
    _align_embedding_model(cfg, trained)

    encoder = load_encoder(cfg, device)
    frame = pd.DataFrame()
    if args.file:
        path = Path(args.file)
        frame = read_table(path)
        column = match_column(frame, args.text_column or cfg.data.text_column, "文本")
        texts = frame[column].fillna("").astype(str).str.strip().tolist()
        if not any(texts):
            raise ValueError(f"{path} 里没有可用文本，请检查列名 {column}")
    else:
        texts = [args.text]

    vectors = encode_texts(
        encoder,
        texts,
        cfg.embedding.batch_size,
        " 待预测文本",
        trained["chunk_chars"],
    )
    pred = trained["estimator"].predict(vectors)
    scores = decision_scores(trained["estimator"], vectors, pred)

    if args.file:
        _write_predictions(cfg, trained, frame, texts, pred, scores, args)
    else:
        _print_single(trained, texts[0], pred[0], None if scores is None else scores[0])


def cmd_chat(args: argparse.Namespace) -> None:
    cfg = _config(args)
    device = _device(args, cfg)
    model_path = cfg.artifact_dir / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(
            f"没有找到 {model_path}，请先运行：python main.py train"
        )

    trained = joblib.load(model_path)
    _align_embedding_model(cfg, trained)
    encoder = load_encoder(cfg, device)

    print(f"\n{trained['task']} 交互预测，输入 q 退出。")
    while True:
        text = input("\n请输入文本：").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        if not text:
            print("输入不能为空。")
            continue
        vector = encode_texts(
            encoder,
            [text],
            cfg.embedding.batch_size,
            chunk_chars=trained["chunk_chars"],
        )
        pred = trained["estimator"].predict(vector)
        scores = decision_scores(trained["estimator"], vector, pred)
        _print_single(trained, text, pred[0], None if scores is None else scores[0])


def cmd_embed(args: argparse.Namespace) -> None:
    cfg = _config(args)
    device = _device(args, cfg)
    encoder = load_encoder(cfg, device)
    vector = encode_texts(
        encoder,
        [args.text],
        cfg.embedding.batch_size,
        chunk_chars=cfg.embedding.chunk_chars,
    )
    print(f"文本：{args.text}")
    print(f"向量形状：{vector.shape}")
    print(f"前 10 维：{np.array2string(vector[0, :10], precision=4)}")
    print(f"L2 范数：{np.linalg.norm(vector[0]):.6f}")


def _get_embeddings(
    cfg: Config, bundle: Any, encoder: Any, force: bool
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """按配置签名复用向量缓存；配置变了自动重算。"""
    cache = cfg.artifact_dir / "embeddings.npz"
    signature = _cache_signature(cfg, bundle)

    if cache.exists() and not force:
        data = np.load(cache)
        if str(data["signature"]) == signature:
            print(f"复用向量缓存：{cache}（加 --recompute 可强制重算）")
            X = {s: data[f"X_{s}"] for s in SPLITS if f"X_{s}" in data}
            y = {s: data[f"y_{s}"] for s in SPLITS if f"y_{s}" in data}
            return X, y
        print("向量缓存与当前配置不一致，重新编码。")

    X = {
        split: encode_texts(
            encoder,
            bundle.texts[split],
            cfg.embedding.batch_size,
            split,
            cfg.embedding.chunk_chars,
        )
        for split in SPLITS
        if split in bundle.texts
    }
    y = {split: bundle.labels[split] for split in X}

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        signature=np.asarray(signature),
        **{f"X_{k}": v for k, v in X.items()},
        **{f"y_{k}": v for k, v in y.items()},
    )
    print(f"向量已缓存：{cache}")
    return X, y


def _split_digest(bundle: Any, split: str) -> str:
    """单个划分的文本与标签内容摘要。

    条数相同不代表数据相同：改 seed 重新切分、换 split_column、
    同名文件内容更新，都只有内容摘要能发现。
    """
    digest = hashlib.sha256()
    digest.update("\x00".join(bundle.texts[split]).encode("utf-8"))
    digest.update(np.asarray(bundle.labels[split]).tobytes())
    return digest.hexdigest()[:16]


def _cache_signature(cfg: Config, bundle: Any) -> str:
    """向量缓存签名：embedding 配置 + 每个划分的文本与标签内容摘要。

    X 由文本与 embedding 配置决定，y 由标签内容决定；内容一变摘要必然变，
    其余配置字段（数据来源、列名、种子、采样数等）无需逐一比对。
    """
    parts = [
        cfg.embedding.model,
        str(cfg.embedding.chunk_chars),
        *(
            f"{k}:{len(bundle.texts[k])}:{_split_digest(bundle, k)}"
            for k in sorted(bundle.texts)
        ),
    ]
    return ";".join(parts)


def _print_eval(
    cfg: Config,
    bundle: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, Any],
    result: Any,
) -> None:
    print("\n========== 测试集结果 ==========")
    if cfg.task == "classification":
        print(
            f"Accuracy {metrics['accuracy']:.4f}｜"
            f"Macro-F1 {metrics['f1_macro']:.4f}｜"
            f"Weighted-F1 {metrics['f1_weighted']:.4f}"
        )
        labels = metrics["labels"]
        names = [(bundle.label_names or {}).get(v, str(v)) for v in labels]
        print("\n分类报告：")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=names,
                digits=4,
                zero_division=0,
            )
        )
        print("混淆矩阵（行=真实，列=预测，顺序同上）：")
        print(np.asarray(metrics["confusion_matrix"]))
    else:
        print(
            f"R² {metrics['r2']:.4f}｜RMSE {metrics['rmse']:.4f}｜"
            f"MAE {metrics['mae']:.4f}"
        )
        print(f"Pearson {metrics['pearson']:.4f}｜Spearman {metrics['spearman']:.4f}")


def _refit_full(
    cfg: Config,
    result: Any,
    X: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
) -> Any:
    """用最佳参数在全部数据上重训，只作为交付模型，不参与指标计算。"""
    estimator = build_estimator(cfg.task, cfg.model.name, cfg.runtime.seed)
    estimator.set_params(**result.best_params)
    splits = [split for split in SPLITS if split in X]
    estimator.fit(
        np.concatenate([X[split] for split in splits]),
        np.concatenate([y[split] for split in splits]),
    )
    return estimator


def _save_artifacts(
    cfg: Config,
    bundle: Any,
    X: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    result: Any,
    metrics: dict[str, Any],
    y_pred: np.ndarray,
    final_estimator: Any,
    args: argparse.Namespace,
) -> None:
    out = cfg.artifact_dir
    out.mkdir(parents=True, exist_ok=True)
    embedding_dim = int(X["train"].shape[1])

    joblib.dump(
        {
            "task": cfg.task,
            "estimator": final_estimator,
            "embedding_model": cfg.embedding.model,
            "embedding_dim": embedding_dim,
            "chunk_chars": cfg.embedding.chunk_chars,
            "model_name": cfg.model.name,
            "best_params": result.best_params,
            "refit_full": cfg.model.refit_full,
            "label_names": {int(k): v for k, v in (bundle.label_names or {}).items()},
            "metrics": metrics,
        },
        out / MODEL_FILE,
    )

    report = {
        "task": cfg.task,
        "model": cfg.model.name,
        "best_params": result.best_params,
        "embedding_model": cfg.embedding.model,
        "embedding_dim": embedding_dim,
        "dataset": {
            "source": cfg.data.source,
            "name": cfg.data.name,
            "path": cfg.data.path,
            "location": bundle.location,
        },
        "splits": {k: len(v) for k, v in y.items()},
        "refit_full": cfg.model.refit_full,
        "metrics": metrics,
    }
    (out / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    predictions = pd.DataFrame(
        {"text": bundle.texts["test"], "y_true": y["test"], "y_pred": y_pred}
    )
    scores = decision_scores(result.estimator, X["test"], y_pred)
    if scores is not None:
        predictions["score"] = scores
    if cfg.task == "classification" and bundle.label_names:
        names = {int(k): v for k, v in bundle.label_names.items()}
        predictions["y_true_name"] = [names[int(v)] for v in y["test"]]
        predictions["y_pred_name"] = [names[int(v)] for v in y_pred]
    predictions.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")

    shutil.copyfile(Path(args.config), out / "config.yaml")
    print(f"\n模型：{out / MODEL_FILE}")
    print(f"指标：{out / 'metrics.json'}")
    print(f"测试集预测：{out / 'test_predictions.csv'}")


def _write_predictions(
    cfg: Config,
    trained: dict[str, Any],
    frame: pd.DataFrame,
    texts: list[str],
    pred: np.ndarray,
    scores: np.ndarray | None,
    args: argparse.Namespace,
) -> None:
    out = pd.DataFrame({"text": texts})
    if trained["task"] == "classification":
        names = {int(k): v for k, v in trained["label_names"].items()}
        out["pred_id"] = pred.astype(int)
        out["pred"] = [names.get(int(v), str(v)) for v in pred]
    else:
        out["pred"] = pred.astype(float)
    if scores is not None:
        out["score"] = scores

    try:
        label_col = match_column(frame, cfg.data.label_column, "标签")
    except KeyError:
        label_col = None
    if label_col is not None:
        out["y_true"] = frame[label_col].to_numpy()

    target = Path(args.out) if args.out else cfg.artifact_dir / "predictions.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False, encoding="utf-8-sig")

    print(f"共预测 {len(out)} 条，已写出：{target}")
    print(out.head(5).to_string(index=False))


def _print_single(
    trained: dict[str, Any], text: str, pred: Any, score: float | None
) -> None:
    print(f"\n文本：{text}")
    if trained["task"] == "classification":
        names = {int(k): v for k, v in trained["label_names"].items()}
        print(f"预测：{names.get(int(pred), pred)}")
        if score is not None:
            print(f"decision function：{score:+.4f}（有符号距离，不是概率）")
    else:
        print(f"预测值：{float(pred):.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="文本向量 + 浅层模型流水线（分类 / 回归）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, with_device: bool = True) -> None:
        p.add_argument(
            "-c", "--config", default="configs/classification.yaml", help="配置文件"
        )
        if with_device:
            p.add_argument(
                "--device", choices=list(DEVICES), default=None, help="覆盖配置里的设备"
            )

    prepare_parser = subparsers.add_parser("prepare", help="下载数据集与模型到项目目录")
    common(prepare_parser, with_device=False)
    prepare_parser.set_defaults(func=cmd_prepare)

    train_parser = subparsers.add_parser("train", help="编码、调参、训练、评估")
    common(train_parser)
    train_parser.add_argument(
        "--limit", type=int, default=None, help="每个划分只用前 N 条"
    )
    train_parser.add_argument("--recompute", action="store_true", help="忽略向量缓存")
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser("predict", help="单条或批量预测")
    common(predict_parser)
    predict_parser.add_argument("--text", default=None, help="单条文本")
    predict_parser.add_argument(
        "--file", type=Path, default=None, help="批量文件：csv/tsv/parquet/jsonl/json"
    )
    predict_parser.add_argument(
        "--out", type=Path, default=None, help="批量结果输出路径"
    )
    predict_parser.add_argument(
        "--text-column", default=None, help="批量文件的文本列名"
    )
    predict_parser.set_defaults(func=cmd_predict)

    chat_parser = subparsers.add_parser("chat", help="交互预测")
    common(chat_parser)
    chat_parser.set_defaults(func=cmd_chat)

    embed_parser = subparsers.add_parser("embed", help="查看单条文本的句向量")
    common(embed_parser)
    embed_parser.add_argument("text", help="需要编码的文本")
    embed_parser.set_defaults(func=cmd_embed)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from None


if __name__ == "__main__":
    main()
