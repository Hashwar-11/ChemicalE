import json
import math
import sys
import time
from pathlib import Path
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODEL_DIR.parent.parent
sys.path.insert(0, str(TRAINING_DIR))

from pipeline_utils import (
    compute_epoch_metrics,
    finalize_and_save_best,
    load_splits_From_disk,
    print_banner,
    print_epoch_row,
)

CHECKPOINT_PATH = MODEL_DIR / "isolation_forest_checkpoint.npz"


def _c_factor(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    h = math.log(n - 1.0) + 0.5772156649
    return 2.0 * h - (2.0 * (n - 1.0) / n)


def _build_tree(X_sub: np.ndarray, depth: int, max_depth: int, rng: np.random.Generator) -> dict:
    n_samples, n_features = X_sub.shape
    if depth >= max_depth or n_samples <= 1:
        return {"leaf": True, "size": n_samples, "depth": depth}

    feat = int(rng.integers(0, n_features))
    col = X_sub[:, feat]
    vmin, vmax = float(np.min(col)), float(np.max(col))
    if vmin == vmax:
        return {"leaf": True, "size": n_samples, "depth": depth}

    split = float(rng.uniform(vmin, vmax))
    left_mask = col < split
    if not np.any(left_mask) or np.all(left_mask):
        return {"leaf": True, "size": n_samples, "depth": depth}

    return {
        "leaf": False,
        "feat": feat,
        "split": split,
        "left": _build_tree(X_sub[left_mask], depth + 1, max_depth, rng),
        "right": _build_tree(X_sub[~left_mask], depth + 1, max_depth, rng),
    }


def _eval_tree(node: dict, X: np.ndarray, idx: np.ndarray, out_lengths: np.ndarray) -> None:
    if len(idx) == 0:
        return
    if node["leaf"]:
        out_lengths[idx] = node["depth"] + _c_factor(node["size"])
        return
    col = X[idx, node["feat"]]
    left_mask = col < node["split"]
    _eval_tree(node["left"], X, idx[left_mask], out_lengths)
    _eval_tree(node["right"], X, idx[~left_mask], out_lengths)


def run_evaluation() -> dict:
    splits = load_splits_From_disk()
    X_train = splits["X_train_scaled"]
    X_val = splits["X_val_scaled"]
    X_test = splits["X_test_scaled"]

    model_name = "Isolation Forest (Progressive Tree Ensemble)"
    family = "Tree-Based Random Partitioning"

    max_samples = 256
    max_depth = int(math.ceil(math.log2(max_samples)))
    c_norm = _c_factor(max_samples)

    epochs_to_run = 10
    trees_per_epoch = 15
    trees: list[dict] = []

    sum_len_train = np.zeros(len(X_train), dtype=np.float64)
    sum_len_val = np.zeros(len(X_val), dtype=np.float64)
    sum_len_test = np.zeros(len(X_test), dtype=np.float64)

    idx_train = np.arange(len(X_train), dtype=np.int64)
    idx_val = np.arange(len(X_val), dtype=np.int64)
    idx_test = np.arange(len(X_test), dtype=np.int64)

    buf_train = np.empty(len(X_train), dtype=np.float64)
    buf_val = np.empty(len(X_val), dtype=np.float64)
    buf_test = np.empty(len(X_test), dtype=np.float64)

    best_score = -1.0
    best_eval = None
    best_epoch_label = ""
    best_train_metric = ""
    best_scores_test = None
    best_infer_ms = 0.0
    start_epoch = 1
    resume_msg = None
    loaded_m = None

    if CHECKPOINT_PATH.exists():
        ckpt = np.load(CHECKPOINT_PATH, allow_pickle=False)
        trees = json.loads(str(ckpt["trees_json"]))
        prev_epoch = int(ckpt["best_epoch"]) if "best_epoch" in ckpt else (len(trees) // trees_per_epoch)
        start_epoch = prev_epoch + 1

        t0 = time.perf_counter()
        for tree in trees:
            _eval_tree(tree, X_train, idx_train, buf_train)
            sum_len_train += buf_train
            _eval_tree(tree, X_val, idx_val, buf_val)
            sum_len_val += buf_val
            _eval_tree(tree, X_test, idx_test, buf_test)
            sum_len_test += buf_test
        infer_ms = (time.perf_counter() - t0) * 1000.0

        n_t = len(trees)
        scores_tr = np.power(2.0, -(sum_len_train / n_t) / c_norm)
        scores_va = np.power(2.0, -(sum_len_val / n_t) / c_norm)
        scores_te = np.power(2.0, -(sum_len_test / n_t) / c_norm)
        thresh = float(np.quantile(scores_tr, 0.99))

        loaded_m = compute_epoch_metrics(
            thresh,
            scores_tr,
            scores_va,
            scores_te,
            0.78,
            splits,
            loss_train=scores_tr ** 2,
            loss_val=scores_va ** 2,
            loss_test=scores_te ** 2,
        )
        best_score = loaded_m["composite_score"]
        best_eval = loaded_m
        best_epoch_label = f"Saved Epoch {prev_epoch:02d} ({n_t:3d} Trees)"
        best_train_metric = f"99%UCL={thresh:.4f}"
        best_scores_test = scores_te.copy()
        best_infer_ms = infer_ms
        resume_msg = (
            f"Loaded {CHECKPOINT_PATH.name} -> {best_epoch_label}, "
            f"{best_train_metric}, Best Score={best_score:.2f}/100 (Resuming at Epoch {start_epoch:02d})"
        )

    print_banner(model_name, family, splits, resume_msg=resume_msg)

    if loaded_m is not None:
        print_epoch_row(
            best_epoch_label,
            best_train_metric,
            loaded_m,
            is_best=False,
            status_override="✓ LOADED BEST CHECKPOINT",
        )

    rng = np.random.default_rng(42 + len(trees))
    end_epoch = start_epoch + epochs_to_run - 1
    epoch_history = []

    for ep in range(start_epoch, end_epoch + 1):
        t0 = time.perf_counter()
        for _ in range(trees_per_epoch):
            sample_idx = rng.choice(len(X_train), size=max_samples, replace=False)
            tree = _build_tree(X_train[sample_idx], 0, max_depth, rng)
            trees.append(tree)

            _eval_tree(tree, X_train, idx_train, buf_train)
            sum_len_train += buf_train

            _eval_tree(tree, X_val, idx_val, buf_val)
            sum_len_val += buf_val

            _eval_tree(tree, X_test, idx_test, buf_test)
            sum_len_test += buf_test

        n_trees = len(trees)
        scores_train = np.power(2.0, -(sum_len_train / n_trees) / c_norm)
        scores_val = np.power(2.0, -(sum_len_val / n_trees) / c_norm)
        scores_test = np.power(2.0, -(sum_len_test / n_trees) / c_norm)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        threshold = float(np.quantile(scores_train, 0.99))

        m = compute_epoch_metrics(
            threshold=threshold,
            scores_train=scores_train,
            scores_val=scores_val,
            scores_test=scores_test,
            interpretable_weight=0.78,
            splits=splits,
            loss_train=scores_train ** 2,
            loss_val=scores_val ** 2,
            loss_test=scores_test ** 2,
        )

        ep_label = f"Epoch {ep:02d}/{end_epoch:02d} ({n_trees:3d} Trees)"
        tr_metric = f"99%UCL={threshold:.4f}"
        is_best = m["composite_score"] > best_score + 1e-6

        if is_best:
            best_score = m["composite_score"]
            best_eval = m
            best_epoch_label = ep_label
            best_train_metric = tr_metric
            best_scores_test = scores_test.copy()
            best_infer_ms = infer_ms
            np.savez_compressed(
                CHECKPOINT_PATH,
                model_type="Isolation_Forest",
                best_epoch=ep,
                n_estimators=n_trees,
                max_samples=max_samples,
                c_norm=c_norm,
                trees_json=np.array(json.dumps(trees)),
                score_threshold=threshold,
                composite_score=best_score,
                **splits["common_meta"],
            )

        print_epoch_row(ep_label, tr_metric, m, is_best)
        epoch_history.append({
            "epoch": ep,
            "n_trees": n_trees,
            "threshold_99pct": round(threshold, 4),
            "train_cost_mse": m["train_loss_metrics"]["cost_function_mse"],
            "train_rmse": m["train_loss_metrics"]["rmse"],
            "val_rmse": m["val_loss_metrics"]["rmse"],
            "test_rmse": m["test_loss_metrics"]["rmse"],
            "val_roc_auc": round(m["val_roc_auc"], 4),
            "val_f1": m["val_bm"]["f1_score"],
            "test_roc_auc": round(m["test_roc_auc"], 4),
            "test_f1": m["test_bm"]["f1_score"],
            "composite_score": round(m["composite_score"], 2),
            "is_best": is_best,
        })

    return finalize_and_save_best(
        model_key="isolation_forest",
        model_name=model_name,
        family=family,
        model_dir=MODEL_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        best_epoch_label=best_epoch_label,
        best_train_metric=best_train_metric,
        best_eval=best_eval,
        best_scores_test=best_scores_test,
        infer_ms_test=best_infer_ms,
        epoch_history=epoch_history,
        splits=splits,
    )


if __name__ == "__main__":
    run_evaluation()
