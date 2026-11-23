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

CHECKPOINT_PATH = MODEL_DIR / "lof_checkpoint.npz"


def run_evaluation() -> dict:
    splits = load_splits_From_disk()
    X_train = splits["X_train_scaled"]
    X_val = splits["X_val_scaled"]
    X_test = splits["X_test_scaled"]

    model_name = "Local Outlier Factor (LOF Novelty Detector)"
    family = "Local Density k-NN Anomaly Detection"
    print_banner(model_name, family, splits)

    ref_size = 3024
    stride = max(1, len(X_train) // ref_size)
    X_ref = X_train[::stride][:ref_size].copy()
    m_ref = len(X_ref)
    ref_sq_norms = np.sum(X_ref ** 2, axis=1)

    d2_ref = np.maximum(
        0.0,
        ref_sq_norms[:, None] + ref_sq_norms[None, :] - 2.0 * (X_ref @ X_ref.T),
    )
    np.fill_diagonal(d2_ref, np.inf)
    dists_ref = np.sqrt(d2_ref)

    # Precompute sorted top-50 neighbors for train, val, test against X_ref so 10 k-stages run fast
    max_k = 50
    ref_top_idx = np.argpartition(dists_ref, kth=max_k - 1, axis=1)[:, :max_k]
    ref_row = np.arange(m_ref)[:, None]
    ref_top_dists = dists_ref[ref_row, ref_top_idx]
    ref_order = np.argsort(ref_top_dists, axis=1)
    ref_top_idx = ref_top_idx[ref_row, ref_order]
    ref_top_dists = ref_top_dists[ref_row, ref_order]

    def precompute_knn(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(X)
        top_idx = np.empty((n, max_k), dtype=np.int32)
        top_dists = np.empty((n, max_k), dtype=np.float64)
        bs = 2048
        for start in range(0, n, bs):
            Xb = X[start : start + bs]
            xb_norms = np.sum(Xb ** 2, axis=1)
            d2 = np.maximum(0.0, xb_norms[:, None] + ref_sq_norms[None, :] - 2.0 * (Xb @ X_ref.T))
            dists = np.sqrt(d2)
            idx_k = np.argpartition(dists, kth=max_k - 1, axis=1)[:, :max_k]
            r_idx = np.arange(len(Xb))[:, None]
            d_k = dists[r_idx, idx_k]
            ord_k = np.argsort(d_k, axis=1)
            top_idx[start : start + bs] = idx_k[r_idx, ord_k]
            top_dists[start : start + bs] = d_k[r_idx, ord_k]
        return top_idx, top_dists

    tr_idx, tr_dists = precompute_knn(X_train)
    va_idx, va_dists = precompute_knn(X_val)
    te_idx, te_dists = precompute_knn(X_test)

    k_schedule = [5, 8, 12, 16, 20, 25, 30, 35, 40, 50]

    best_score = -1.0
    best_eval = None
    best_epoch_label = ""
    best_train_metric = ""
    best_scores_test = None
    best_infer_ms = 0.0
    epoch_history = []

    for ep_idx, k in enumerate(k_schedule, start=1):
        knn_ref_idx = ref_top_idx[:, :k]
        knn_ref_dists = ref_top_dists[:, :k]
        k_dist_ref = knn_ref_dists[:, -1]
        reach_ref = np.maximum(knn_ref_dists, k_dist_ref[knn_ref_idx])
        lrd_ref = 1.0 / (np.mean(reach_ref, axis=1) + 1e-8)

        def lof_from_precomputed(idx_mat: np.ndarray, dist_mat: np.ndarray) -> np.ndarray:
            ki = idx_mat[:, :k]
            kd = dist_mat[:, :k]
            reach = np.maximum(kd, k_dist_ref[ki])
            lrd_x = 1.0 / (np.mean(reach, axis=1) + 1e-8)
            return np.mean(lrd_ref[ki], axis=1) / lrd_x

        scores_train = lof_from_precomputed(tr_idx, tr_dists)
        threshold = float(np.quantile(scores_train, 0.99))
        scores_val = lof_from_precomputed(va_idx, va_dists)

        t0 = time.perf_counter()
        scores_test = lof_from_precomputed(te_idx, te_dists)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        m = compute_epoch_metrics(
            threshold=threshold,
            scores_train=scores_train,
            scores_val=scores_val,
            scores_test=scores_test,
            interpretable_weight=0.80,
            splits=splits,
        )

        ep_label = f"Epoch {ep_idx:02d}/10 (k={k:02d} NN)"
        tr_metric = f"99%UCL={threshold:.3f}"
        is_best = m["composite_score"] > best_score

        if is_best:
            best_score = m["composite_score"]
            best_eval = m
            best_epoch_label = ep_label
            best_train_metric = tr_metric
            best_scores_test = scores_test
            best_infer_ms = infer_ms
            # Automatically save the new best LOF checkpoint to lof_checkpoint.npz
            np.savez_compressed(
                CHECKPOINT_PATH,
                model_type="Local_Outlier_Factor",
                best_epoch=ep_idx,
                n_neighbors=k,
                X_ref=X_ref,
                k_dist_ref=k_dist_ref,
                lrd_ref=lrd_ref,
                score_threshold=threshold,
                composite_score=best_score,
                **splits["common_meta"],
            )

        print_epoch_row(ep_label, tr_metric, m, is_best)
        epoch_history.append({
            "epoch": ep_idx,
            "n_neighbors": k,
            "threshold_99pct": round(threshold, 4),
            "val_roc_auc": round(m["val_roc_auc"], 4),
            "val_f1": m["val_bm"]["f1_score"],
            "test_roc_auc": round(m["test_roc_auc"], 4),
            "test_f1": m["test_bm"]["f1_score"],
            "composite_score": round(m["composite_score"], 2),
            "is_best": is_best,
        })

    return finalize_and_save_best(
        model_key="local_outlier_factor",
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
