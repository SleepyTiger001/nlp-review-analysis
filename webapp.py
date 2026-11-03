"""本地网页工作台（分类 / 回归）。

先决条件：python main.py train 已生成 artifacts/model.joblib。

用法：
    python webapp.py
    python webapp.py --device cuda --port 8000 --no-browser
    python webapp.py -c configs/regression.yaml

默认只监听 127.0.0.1（可用 --host 改绑）。接口：
    GET  /              工作台页面（templates/index.html）
    GET  /api/model     任务类型、embedding、模型、测试指标、设备、划分规模
    POST /api/predict   {"text": ...} 或 {"vector": [...]} → 类别 / 预测值、向量摘要
    POST /api/neighbors 输入同上 → 测试集中余弦最近的样本
    GET  /api/sample?i= 散点下标 → 该样本的向量、模型输出与原文
    GET  /api/hist      测试集全量的模型输出分布
    GET  /api/space     t-SNE 投影；首次计算几十秒，结果缓存到 artifacts/
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

from config import DEVICES, MODEL_FILE, choose_device, load_config
from data import load_dataset_bundle
from encoder import encode_texts, load_encoder
from modeling import decision_scores

app = Flask(__name__)
app.config.setdefault("DEVICE", "auto")
app.config.setdefault("CONFIG_PATH", "configs/classification.yaml")

_state_lock = threading.Lock()
_space_lock = threading.Lock()
_state: dict[str, Any] = {}


def get_state() -> dict[str, Any]:
    """惰性加载配置、分类器、编码器与测试集全量向量（进程内只加载一次）。"""
    with _state_lock:
        if _state:
            return _state

        cfg = load_config(app.config["CONFIG_PATH"])
        model_file = cfg.artifact_dir / MODEL_FILE
        if not model_file.exists():
            raise RuntimeError("没有找到训练产物，请先运行：python main.py train")

        trained = joblib.load(model_file)

        device = choose_device(app.config["DEVICE"])
        encoder = load_encoder(cfg, device)

        test_set = None
        embeddings_file = cfg.artifact_dir / "embeddings.npz"
        if embeddings_file.exists():
            with np.load(embeddings_file) as data:
                test_set = {"X": data["X_test"], "y": data["y_test"]}

        _state.update(
            cfg=cfg,
            trained=trained,
            encoder=encoder,
            device=device,
            test=test_set,
            texts=None,
            texts_failed=False,
        )
        return _state


def load_test_texts() -> list[str] | None:
    """读取测试集原文（最近邻展示用）；失败后记住，避免反复读盘。"""
    state = get_state()
    if state["texts"] is None and not state["texts_failed"]:
        try:
            bundle = load_dataset_bundle(state["cfg"], verbose=False)
            state["texts"] = bundle.texts["test"]
        except Exception:
            state["texts"], state["texts_failed"] = [], True
    return state["texts"] or None


def encode_one(text: str) -> np.ndarray:
    state = get_state()
    return encode_texts(
        state["encoder"],
        [text],
        state["cfg"].embedding.batch_size,
        chunk_chars=state["trained"]["chunk_chars"],
    )


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/model")
def api_model() -> Any:
    try:
        state = get_state()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    trained = state["trained"]
    splits = {}
    embeddings_file = state["cfg"].artifact_dir / "embeddings.npz"
    if embeddings_file.exists():
        with np.load(embeddings_file) as data:
            splits = {
                k: int(data[f"y_{k}"].shape[0])
                for k in ("train", "validation", "test")
                if f"y_{k}" in data
            }
    return jsonify(
        {
            "task": trained["task"],
            "embedding_model": trained["embedding_model"],
            "embedding_dim": int(trained["embedding_dim"]),
            "classifier": type(trained["estimator"]).__name__,
            "best_params": trained["best_params"],
            "labels": {str(k): v for k, v in trained["label_names"].items()},
            "n_classes": len(trained["label_names"]),
            "device": state["device"],
            "versions": {
                "python": ".".join(map(str, sys.version_info[:3])),
                "torch": torch.__version__,
            },
            "splits": splits,
            "test_n": int(state["test"]["X"].shape[0]) if state["test"] else 0,
            "metrics": trained["metrics"],
        }
    )


def _vec_from_payload(payload: dict[str, Any], state: dict[str, Any]) -> np.ndarray:
    """优先用请求里的现成向量（点选样本场景），否则把 text 编码成向量。

    向量路径约定传入已归一化的 embedding（前端点选样本满足此约定）。
    """
    raw = payload.get("vector")
    if raw is not None:
        try:
            vec = np.asarray(raw, dtype=np.float32).reshape(1, -1)
        except Exception:
            raise ValueError("vector 格式错误") from None
        expected = int(state["trained"]["embedding_dim"])
        if vec.shape[1] != expected:
            raise ValueError(f"vector 维度应为 {expected}")
        return vec
    text = str(payload.get("text", "")).strip()
    if not text:
        raise ValueError("请输入文本")
    return encode_one(text)


@app.post("/api/predict")
def api_predict() -> Any:
    started = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    try:
        state = get_state()
        vec = _vec_from_payload(payload, state)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    encode_ms = round((time.perf_counter() - started) * 1000)

    trained = state["trained"]
    estimator = trained["estimator"]
    task = trained["task"]
    labels = {int(k): v for k, v in trained["label_names"].items()}
    pred = estimator.predict(vec)
    scores = decision_scores(estimator, vec, pred)
    if task == "classification":
        label_id = int(pred[0])
        label = labels.get(label_id, str(label_id))
        value = None
        margin = float(scores[0]) if scores is not None else None
    else:
        label_id = None
        value = float(pred[0])
        label = f"{value:.2f}"
        margin = value

    vector = vec[0]
    top = np.argsort(-np.abs(vector))[:6]
    return jsonify(
        {
            "task": task,
            "label_id": label_id,
            "label": label,
            "value": value,
            "margin": margin,
            "dim": int(vector.shape[0]),
            "norm": float(np.linalg.norm(vector)),
            "top_dims": [{"index": int(i), "value": float(vector[i])} for i in top],
            "vector": [round(float(x), 4) for x in vector],
            "timing": {
                "encode_ms": encode_ms,
                "total_ms": round((time.perf_counter() - started) * 1000),
            },
        }
    )


@app.post("/api/neighbors")
def api_neighbors() -> Any:
    payload = request.get_json(silent=True) or {}
    try:
        state = get_state()
        vec = _vec_from_payload(payload, state)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    sub = state["test"]
    if sub is None:
        return jsonify({"available": False, "reason": "缺少 artifacts/embeddings.npz"})

    labels = {int(k): v for k, v in state["trained"]["label_names"].items()}
    task = state["trained"]["task"]
    query = vec[0]
    sims = sub["X"] @ query  # 双方均已 L2 归一化，点积即余弦
    exclude = payload.get("exclude")
    if isinstance(exclude, int) and 0 <= exclude < len(sims):
        sims[exclude] = -np.inf  # 点选样本时排除原句自身
    texts = load_test_texts()
    items = []
    for pos in np.argsort(-sims)[:3]:
        if task == "classification":
            label_id = int(sub["y"][pos])
            label = labels.get(label_id, str(label_id))
            value = None
        else:
            label_id = None
            value = float(sub["y"][pos])
            label = f"{value:.2f}"
        items.append(
            {
                "scatter_index": int(pos),
                "sim": round(float(sims[pos]), 4),
                "label_id": label_id,
                "label": label,
                "value": value,
                "text": (texts[pos][:120] if texts else None),
            }
        )
    return jsonify(
        {"available": True, "items": items, "text_available": texts is not None}
    )


@app.get("/api/sample")
def api_sample() -> Any:
    """t-SNE 点选：按散点下标返回该测试样本的向量、判定与原文。"""
    try:
        state = get_state()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    sub = state["test"]
    if sub is None:
        return jsonify({"error": "缺少 artifacts/embeddings.npz"}), 503

    index = request.args.get("i", type=int)
    if index is None or not (0 <= index < len(sub["y"])):
        return jsonify({"error": "样本索引越界"}), 400

    vector = sub["X"][index : index + 1]
    trained = state["trained"]
    estimator = trained["estimator"]
    task = trained["task"]
    labels = {int(k): v for k, v in trained["label_names"].items()}
    texts = load_test_texts()
    pred = estimator.predict(vector)
    scores = decision_scores(estimator, vector, pred)
    if task == "classification":
        label_id = int(pred[0])
        label = labels.get(label_id, str(label_id))
        value = None
        margin = float(scores[0]) if scores is not None else None
    else:
        label_id = None
        value = float(pred[0])
        label = f"{value:.2f}"
        margin = value
    true = int(sub["y"][index]) if task == "classification" else float(sub["y"][index])
    return jsonify(
        {
            "task": task,
            "i": index,
            "label_id": label_id,
            "label": label,
            "value": value,
            "margin": margin,
            "true_label": true,
            "text": (texts[index][:120] if texts else None),
            "vector": [round(float(x), 4) for x in vector[0]],
        }
    )


def _test_outputs(state: dict[str, Any]) -> tuple[list[float], list[int] | None]:
    """测试集全量输出，进程内缓存一次。

    分类为「预测类的决策值」（多分类取预测类对应列），回归为预测值；
    preds 仅分类任务有值，供前端按「预测正确 / 错误」着色。
    """
    if "margins" not in state:
        sub = state["test"]
        trained = state["trained"]
        estimator = trained["estimator"]
        pred = np.asarray(estimator.predict(sub["X"]))
        if trained["task"] == "classification":
            scores = decision_scores(estimator, sub["X"], pred)
            margins = scores if scores is not None else pred.astype(float)
            state["preds"] = [int(v) for v in pred]
        else:
            margins = pred.astype(float)
            state["preds"] = None
        state["margins"] = np.asarray(margins, dtype=float).tolist()
    return state["margins"], state["preds"]


@app.get("/api/hist")
def api_hist() -> Any:
    """测试集全量的模型输出分布（分类为决策值，回归为预测值）。"""
    try:
        state = get_state()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    sub = state["test"]
    if sub is None:
        return jsonify({"error": "缺少 artifacts/embeddings.npz"}), 503

    trained = state["trained"]
    margins, preds = _test_outputs(state)
    labels = [float(v) if trained["task"] == "regression" else int(v) for v in sub["y"]]
    return jsonify(
        {"task": trained["task"], "margins": margins, "labels": labels, "preds": preds}
    )


def _space_signature(sub: dict[str, np.ndarray]) -> str:
    """t-SNE 缓存签名：测试集向量与标签的内容摘要，换模型或数据即失效。"""
    digest = hashlib.sha256()
    digest.update(sub["X"].tobytes())
    digest.update(sub["y"].tobytes())
    return digest.hexdigest()


def _load_space_cache(
    cache: Path, sub: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache.exists():
        return None
    with np.load(cache) as cached:
        if str(cached["signature"]) != _space_signature(sub):
            print("t-SNE 缓存与当前测试集不一致，重新计算。")
            return None
        return cached["xy"], cached["y"]


@app.get("/api/space")
def api_space() -> Any:
    try:
        state = get_state()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    sub = state["test"]
    if sub is None:
        return jsonify({"error": "缺少 artifacts/embeddings.npz，无法绘制投影"}), 503

    cache = state["cfg"].artifact_dir / "space_tsne.npz"
    task = state["trained"]["task"]
    _, preds = _test_outputs(state)
    with _space_lock:
        cached_payload = _load_space_cache(cache, sub)
        if cached_payload is not None:
            xy, labels = cached_payload
            return jsonify(
                {
                    "task": task,
                    "points": xy.tolist(),
                    "labels": labels.tolist(),
                    "preds": preds,
                }
            )

        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        seed = state["cfg"].runtime.seed
        print(f"计算 t-SNE 投影（{len(sub['y'])} 个点，首次约需几十秒）…")
        X50 = PCA(n_components=50, random_state=seed).fit_transform(sub["X"])
        xy = (
            TSNE(n_components=2, init="pca", perplexity=30, random_state=seed)
            .fit_transform(X50)
            .astype(np.float32)
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, xy=xy, y=sub["y"], signature=np.asarray(_space_signature(sub))
        )
        print("t-SNE 完成，已缓存。")
    return jsonify(
        {
            "task": task,
            "points": xy.tolist(),
            "labels": sub["y"].tolist(),
            "preds": preds,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="本地预测工作台（分类 / 回归）")
    parser.add_argument("-c", "--config", default="configs/classification.yaml")
    parser.add_argument("--device", choices=list(DEVICES), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=int, default=7860
    )  # 5000 会被 macOS AirPlay 占用
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    app.config["DEVICE"] = args.device
    app.config["CONFIG_PATH"] = args.config
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    print(f"工作台地址：{url}（Ctrl+C 退出）")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
