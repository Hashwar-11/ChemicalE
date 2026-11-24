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

    cov = np.cov(X_train, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-8)
    eigvecs = eigvecs[:, order]
    cumvar = np.cumsum(eigvals) / np.sum(eigvals)
    max_pcs = X_train.shape[1]

    def evaluate_k_components(k: int, t2_weight: float = 1.0) -> tuple[dict, np.ndarray, float, dict]:
        k = int(np.clip(k, 1, max_pcs))
        retained_var = float(cumvar[k - 1])
        comps = eigvecs[:, :k]
        evals = eigvals[:k]

        s_tr = X_train @ comps
        t2_tr = np.sum((s_tr ** 2) / evals, axis=1)
        q_tr = np.sum((X_train - (s_tr @ comps.T)) ** 2, axis=1)
        t2_ucl = float(np.quantile(t2_tr, 0.99))
        q_ucl = float(np.quantile(q_tr, 0.99))

        def score_fn(X: np.ndarray) -> np.ndarray:
            s = X @ comps
            t2 = np.sum((s ** 2) / evals, axis=1)
            xr = s @ comps.T
            q = np.sum((X - xr) ** 2, axis=1)
            return np.maximum(t2_weight * (t2 / t2_ucl), q / q_ucl)

        scores_train = np.maximum(t2_weight * (t2_tr / t2_ucl), q_tr / q_ucl)
        threshold = float(np.quantile(scores_train, 0.99))
        scores_val = score_fn(X_val)

        t0 = time.perf_counter()
        scores_test = score_fn(X_test)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        m = compute_epoch_metrics(
            threshold=threshold,
            scores_train=scores_train,
            scores_val=scores_val,
            scores_test=scores_test,
            interpretable_weight=1.00,
            splits=splits,
        )
        state = {
            "k": k,
            "t2_weight": t2_weight,
            "retained_var": retained_var,
            "comps": comps,
            "evals": evals,
            "t2_ucl": t2_ucl,
            "q_ucl": q_ucl,
            "threshold": threshold,
        }
        return m, scores_test, infer_ms, state

    # Check if a saved best checkpoint already exists on disk and resume from it
    best_score = -1.0
    best_eval = None
    best_epoch_label = ""
    best_train_metric = ""
    best_scores_test = None
    best_infer_ms = 0.0
    start_epoch = 1
    resume_k = None
    resume_w = 1.0
    resume_msg = None
    loaded_m = None

    if CHECKPOINT_PATH.exists():
        ckpt = np.load(CHECKPOINT_PATH, allow_pickle=False)
        resume_k = int(ckpt["n_components"])
        resume_w = float(ckpt["t2_weight"]) if "t2_weight" in ckpt else 1.0
        prev_epoch = int(ckpt["best_epoch"]) if "best_epoch" in ckpt else 9
        start_epoch = prev_epoch + 1

        loaded_m, loaded_scores_test, loaded_infer_ms, loaded_state = evaluate_k_components(resume_k, resume_w)
        best_score = loaded_m["composite_score"]
        best_eval = loaded_m
        best_epoch_label = f"Saved Epoch {prev_epoch:02d} (k={resume_k:02d} PCs)"
        best_train_metric = f"Var={loaded_state['retained_var']*100:5.2f}%"
        best_scores_test = loaded_scores_test
        best_infer_ms = loaded_infer_ms
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

    # If resuming from best k, explore fine neighborhood around best (k, t2_weight); otherwise do initial sweep
    if resume_k is not None:
        candidates = [
            (max(1, resume_k - 2), resume_w),
            (max(1, resume_k - 1), resume_w),
            (resume_k, round(max(0.5, resume_w - 0.10), 2)),
            (resume_k, round(max(0.5, resume_w - 0.05), 2)),
            (resume_k, resume_w),
            (resume_k, round(resume_w + 0.05, 2)),
            (resume_k, round(resume_w + 0.10, 2)),
            (min(max_pcs, resume_k + 1), resume_w),
            (min(max_pcs, resume_k + 2), resume_w),
            (min(max_pcs, resume_k + 3), resume_w),
        ]
    else:
        variance_schedule = [0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.96, 0.97, 0.98]
        candidates = [(min(int(np.searchsorted(cumvar, v) + 1), max_pcs), 1.0) for v in variance_schedule]

    epoch_history = []
    end_epoch = start_epoch + len(candidates) - 1

    for offset, (k_cand, w_cand) in enumerate(candidates):
        ep_num = start_epoch + offset
        m, scores_test, infer_ms, st = evaluate_k_components(k_cand, w_cand)

        ep_label = f"Epoch {ep_num:02d}/{end_epoch:02d} (k={st['k']:02d},w={w_cand:.2f})"
        tr_metric = f"Var={st['retained_var']*100:5.2f}%"
        is_best = m["composite_score"] > best_score + 1e-6

        if is_best:
            best_score = m["composite_score"]
            best_eval = m
            best_epoch_label = ep_label
            best_train_metric = tr_metric
            best_scores_test = scores_test
            best_infer_ms = infer_ms
            np.savez_compressed(
                CHECKPOINT_PATH,
                model_type="PCA_T2_Q",
                best_epoch=ep_num,
                components=st["comps"],
                eigenvalues=st["evals"],
                n_components=st["k"],
                t2_weight=st["t2_weight"],
                retained_variance=st["retained_var"],
                t2_ucl=st["t2_ucl"],
                q_ucl=st["q_ucl"],
                score_threshold=st["threshold"],
                composite_score=best_score,
                **splits["common_meta"],
            )

        print_epoch_row(ep_label, tr_metric, m, is_best)
        epoch_history.append({
            "epoch": ep_num,
            "n_components": st["k"],
            "t2_weight": w_cand,
            "retained_variance": round(st["retained_var"], 4),
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
