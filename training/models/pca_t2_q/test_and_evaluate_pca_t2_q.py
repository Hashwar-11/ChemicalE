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

CHECKPOINT_PATH = MODEL_DIR / "pca_t2_q_checkpoint.npz"


def run_evaluation() -> dict:
    splits = load_splits_From_disk()
    X_train = splits["X_train_scaled"]
    X_val = splits["X_val_scaled"]
    X_test = splits["X_test_scaled"]

    model_name = "PCA Multivariate Monitor (Hotelling T² + Q-Stat)"
    family = "Subspace Projection & Residual SPC"
    print_banner(model_name, family, splits)

    cov = np.cov(X_train, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-8)
    eigvecs = eigvecs[:, order]
    cumvar = np.cumsum(eigvals) / np.sum(eigvals)

    # 10 progressive subspace epochs sweeping variance retained from 75% to 98%
    variance_schedule = [0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.96, 0.97, 0.98]

    best_score = -1.0
    best_eval = None
    best_epoch_label = ""
    best_train_metric = ""
    best_scores_test = None
    best_infer_ms = 0.0
    epoch_history = []

    for ep_idx, var_target in enumerate(variance_schedule, start=1):
        k = int(np.searchsorted(cumvar, var_target) + 1)
        k = min(k, X_train.shape[1])
        retained_var = float(cumvar[k - 1])
        comps = eigvecs[:, :k]
        evals = eigvals[:k]

        def score_fn(X: np.ndarray, t2_u: float, q_u: float) -> np.ndarray:
            s = X @ comps
            t2 = np.sum((s ** 2) / evals, axis=1)
            xr = s @ comps.T
            q = np.sum((X - xr) ** 2, axis=1)
            return np.maximum(t2 / t2_u, q / q_u)

        s_tr = X_train @ comps
        t2_tr = np.sum((s_tr ** 2) / evals, axis=1)
        q_tr = np.sum((X_train - (s_tr @ comps.T)) ** 2, axis=1)
        t2_ucl = float(np.quantile(t2_tr, 0.99))
        q_ucl = float(np.quantile(q_tr, 0.99))

        scores_train = np.maximum(t2_tr / t2_ucl, q_tr / q_ucl)
        threshold = float(np.quantile(scores_train, 0.99))
        scores_val = score_fn(X_val, t2_ucl, q_ucl)

        t0 = time.perf_counter()
        scores_test = score_fn(X_test, t2_ucl, q_ucl)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        m = compute_epoch_metrics(
            threshold=threshold,
            scores_train=scores_train,
            scores_val=scores_val,
            scores_test=scores_test,
            interpretable_weight=1.00,
            splits=splits,
        )

        ep_label = f"Epoch {ep_idx:02d}/10 (k={k:02d} PCs)"
        tr_metric = f"Var={retained_var*100:5.2f}%"
        is_best = m["composite_score"] > best_score

        if is_best:
            best_score = m["composite_score"]
            best_eval = m
            best_epoch_label = ep_label
            best_train_metric = tr_metric
            best_scores_test = scores_test
            best_infer_ms = infer_ms
            # Automatically save the new best checkpoint to pca_t2_q_checkpoint.npz
            np.savez_compressed(
                CHECKPOINT_PATH,
                model_type="PCA_T2_Q",
                best_epoch=ep_idx,
                components=comps,
                eigenvalues=evals,
                n_components=k,
                retained_variance=retained_var,
                t2_ucl=t2_ucl,
                q_ucl=q_ucl,
                score_threshold=threshold,
                composite_score=best_score,
                **splits["common_meta"],
            )

        print_epoch_row(ep_label, tr_metric, m, is_best)
        epoch_history.append({
            "epoch": ep_idx,
            "n_components": k,
            "retained_variance": round(retained_var, 4),
            "val_roc_auc": round(m["val_roc_auc"], 4),
            "val_f1": m["val_bm"]["f1_score"],
            "test_roc_auc": round(m["test_roc_auc"], 4),
            "test_f1": m["test_bm"]["f1_score"],
            "composite_score": round(m["composite_score"], 2),
            "is_best": is_best,
        })

    return finalize_and_save_best(
        model_key="pca_t2_q",
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
