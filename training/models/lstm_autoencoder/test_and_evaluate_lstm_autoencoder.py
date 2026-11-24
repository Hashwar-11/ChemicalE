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

CHECKPOINT_PATH = MODEL_DIR / "lstm_autoencoder_checkpoint.npz"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def _make_windows(X: np.ndarray, seq_len: int, stride: int = 1) -> np.ndarray:
    _, d = X.shape
    X_pad = np.vstack([np.repeat(X[:1], seq_len - 1, axis=0), X])
    s0, s1 = X_pad.strides
    n_win = (len(X_pad) - seq_len) // stride + 1
    return np.lib.stride_tricks.as_strided(
        X_pad,
        shape=(n_win, seq_len, d),
        strides=(s0 * stride, s0, s1),
    )


def _forward_lstm_batch(
    W_seq: np.ndarray,
    W_enc: np.ndarray,
    U_enc: np.ndarray,
    b_enc: np.ndarray,
    W_ctx: np.ndarray,
    b_ctx: np.ndarray,
    W_dec: np.ndarray,
    b_dec: np.ndarray,
    H: int,
) -> tuple[np.ndarray, dict]:
    B, L, _ = W_seq.shape
    h = np.zeros((B, H), dtype=np.float64)
    c = np.zeros((B, H), dtype=np.float64)
    h_sum = np.zeros((B, H), dtype=np.float64)
    gates_cache = []

    for t in range(L):
        xt = W_seq[:, t, :]
        z = xt @ W_enc + h @ U_enc + b_enc
        f = _sigmoid(z[:, :H])
        i = _sigmoid(z[:, H : 2 * H])
        g = np.tanh(np.clip(z[:, 2 * H : 3 * H], -15.0, 15.0))
        o = _sigmoid(z[:, 3 * H :])

        c_next = f * c + i * g
        tanh_c = np.tanh(np.clip(c_next, -15.0, 15.0))
        h_next = o * tanh_c

        gates_cache.append((xt, h, c, f, i, g, o, c_next, tanh_c))
        h = h_next
        c = c_next
        h_sum += h

    ctx_in = np.hstack([h, h_sum / L])
    latent = np.tanh(ctx_in @ W_ctx + b_ctx)
    x_hat = latent @ W_dec + b_dec
    return x_hat, {"gates": gates_cache, "ctx_in": ctx_in, "latent": latent}


def _score_dataset(
    X: np.ndarray,
    seq_len: int,
    H: int,
    W_enc: np.ndarray,
    U_enc: np.ndarray,
    b_enc: np.ndarray,
    W_ctx: np.ndarray,
    b_ctx: np.ndarray,
    W_dec: np.ndarray,
    b_dec: np.ndarray,
) -> np.ndarray:
    wins = _make_windows(X, seq_len=seq_len, stride=1)
    n = len(wins)
    scores = np.empty(n, dtype=np.float64)
    bs = 2048
    for start in range(0, n, bs):
        W_b = wins[start : start + bs]
        x_hat, _ = _forward_lstm_batch(W_b, W_enc, U_enc, b_enc, W_ctx, b_ctx, W_dec, b_dec, H)
        target = W_b[:, -1, :]
        scores[start : start + bs] = np.mean((target - x_hat) ** 2, axis=1)
    return scores


def run_evaluation() -> dict:
    splits = load_splits_From_disk()
    X_train = splits["X_train_scaled"]
    X_val = splits["X_val_scaled"]
    X_test = splits["X_test_scaled"]

    model_name = "LSTM Autoencoder (SeqLen=10, Hidden=20)"
    family = "Deep Temporal Sequence Reconstruction"

    seq_len = 10
    H = 20
    epochs_to_run = 10
    batch_size = 256
    _, D = X_train.shape

    best_score = -1.0
    best_eval = None
    best_epoch_label = ""
    best_train_metric = ""
    best_scores_test = None
    best_infer_ms = 0.0
    start_epoch = 1
    loss_history: list[float] = []
    resume_msg = None
    loaded_m = None

    if CHECKPOINT_PATH.exists():
        ckpt = np.load(CHECKPOINT_PATH, allow_pickle=False)
        W_enc = ckpt["W_enc"].copy()
        U_enc = ckpt["U_enc"].copy()
        b_enc = ckpt["b_enc"].copy()
        W_ctx = ckpt["W_ctx"].copy()
        b_ctx = ckpt["b_ctx"].copy()
        W_dec = ckpt["W_dec"].copy()
        b_dec = ckpt["b_dec"].copy()
        loss_history = [float(x) for x in ckpt["loss_history"]] if "loss_history" in ckpt else [0.07249]
        prev_epoch = int(ckpt["best_epoch"]) if "best_epoch" in ckpt else len(loss_history)
        start_epoch = prev_epoch + 1
        lr = 0.002  # Fine-tuning learning rate when resuming from saved best weights

        scores_tr = _score_dataset(X_train, seq_len, H, W_enc, U_enc, b_enc, W_ctx, b_ctx, W_dec, b_dec)
        thresh = float(np.quantile(scores_tr, 0.99))
        scores_va = _score_dataset(X_val, seq_len, H, W_enc, U_enc, b_enc, W_ctx, b_ctx, W_dec, b_dec)
        t0 = time.perf_counter()
        scores_te = _score_dataset(X_test, seq_len, H, W_enc, U_enc, b_enc, W_ctx, b_ctx, W_dec, b_dec)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        loaded_m = compute_epoch_metrics(thresh, scores_tr, scores_va, scores_te, 0.90, splits)
        best_score = loaded_m["composite_score"]
        best_eval = loaded_m
        best_epoch_label = f"Saved Epoch {prev_epoch:02d}"
        best_train_metric = f"MSE={loss_history[-1]:.5f}"
        best_scores_test = scores_te
        best_infer_ms = infer_ms
        resume_msg = (
            f"Loaded {CHECKPOINT_PATH.name} -> {best_epoch_label} ({best_train_metric}), "
            f"Best Score={best_score:.2f}/100 (Resuming at Epoch {start_epoch:02d})"
        )
    else:
        rng_init = np.random.default_rng(42)
        W_enc = rng_init.normal(0.0, math.sqrt(2.0 / (D + H)), size=(D, 4 * H))
        U_enc = rng_init.normal(0.0, math.sqrt(1.0 / H), size=(H, 4 * H))
        b_enc = np.zeros(4 * H, dtype=np.float64)
        b_enc[:H] = 1.0
        W_ctx = rng_init.normal(0.0, math.sqrt(2.0 / (3 * H)), size=(2 * H, H))
        b_ctx = np.zeros(H, dtype=np.float64)
        W_dec = rng_init.normal(0.0, math.sqrt(2.0 / (H + D)), size=(H, D))
        b_dec = np.zeros(D, dtype=np.float64)
        lr = 0.008

    print_banner(model_name, family, splits, resume_msg=resume_msg)

    if loaded_m is not None:
        print_epoch_row(
            best_epoch_label,
            best_train_metric,
            loaded_m,
            is_best=False,
            status_override="✓ LOADED BEST CHECKPOINT",
        )

    train_wins = _make_windows(X_train, seq_len=seq_len, stride=3)
    n_wins = len(train_wins)

    params = [W_enc, U_enc, b_enc, W_ctx, b_ctx, W_dec, b_dec]
    m_adam = [np.zeros_like(p) for p in params]
    v_adam = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    rng = np.random.default_rng(42 + start_epoch)
    end_epoch = start_epoch + epochs_to_run - 1
    epoch_history = []

    for ep in range(start_epoch, end_epoch + 1):
        perm = rng.permutation(n_wins)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_wins, batch_size):
            batch_idx = perm[start : start + batch_size]
            W_b = train_wins[batch_idx]
            B = len(W_b)
            target = W_b[:, -1, :]

            x_hat, cache = _forward_lstm_batch(W_b, W_enc, U_enc, b_enc, W_ctx, b_ctx, W_dec, b_dec, H)
            err = x_hat - target
            epoch_loss += float(np.mean(err ** 2))
            n_batches += 1

            d_xhat = (2.0 / (B * D)) * err
            latent = cache["latent"]
            ctx_in = cache["ctx_in"]

            dW_dec = latent.T @ d_xhat
            db_dec = np.sum(d_xhat, axis=0)
            d_latent = (d_xhat @ W_dec.T) * (1.0 - latent ** 2)
            dW_ctx = ctx_in.T @ d_latent
            db_ctx = np.sum(d_latent, axis=0)

            d_ctx_in = d_latent @ W_ctx.T
            dh_last = d_ctx_in[:, :H]
            dh_mean = d_ctx_in[:, H:] / seq_len

            dW_enc = np.zeros_like(W_enc)
            dU_enc = np.zeros_like(U_enc)
            db_enc = np.zeros_like(b_enc)

            dh_next = dh_last.copy()
            dc_next = np.zeros((B, H), dtype=np.float64)
            gates = cache["gates"]
            for t in range(seq_len - 1, max(-1, seq_len - 5), -1):
                xt, h_prev, c_prev, f, i, g, o, _, tanh_c = gates[t]
                dh = dh_next + dh_mean
                do = dh * tanh_c * o * (1.0 - o)
                dc = dh * o * (1.0 - tanh_c ** 2) + dc_next
                df = dc * c_prev * f * (1.0 - f)
                di = dc * g * i * (1.0 - i)
                dg = dc * i * (1.0 - g ** 2)

                dz = np.hstack([df, di, dg, do])
                dW_enc += xt.T @ dz
                dU_enc += h_prev.T @ dz
                db_enc += np.sum(dz, axis=0)

                dh_next = dz @ U_enc.T
                dc_next = dc * f

            grads = [dW_enc, dU_enc, db_enc, dW_ctx, db_ctx, dW_dec, db_dec]
            step += 1
            for idx_p, (p, g_p) in enumerate(zip(params, grads)):
                g_clip = np.clip(g_p, -1.0, 1.0)
                m_adam[idx_p] = beta1 * m_adam[idx_p] + (1.0 - beta1) * g_clip
                v_adam[idx_p] = beta2 * v_adam[idx_p] + (1.0 - beta2) * (g_clip ** 2)
                m_hat = m_adam[idx_p] / (1.0 - beta1 ** step)
                v_hat = v_adam[idx_p] / (1.0 - beta2 ** step)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        mean_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(round(mean_loss, 5))

        scores_train = _score_dataset(X_train, seq_len, H, W_enc, U_enc, b_enc, W_ctx, b_ctx, W_dec, b_dec)
        threshold = float(np.quantile(scores_train, 0.99))
        scores_val = _score_dataset(X_val, seq_len, H, W_enc, U_enc, b_enc, W_ctx, b_ctx, W_dec, b_dec)

        t0 = time.perf_counter()
        scores_test = _score_dataset(X_test, seq_len, H, W_enc, U_enc, b_enc, W_ctx, b_ctx, W_dec, b_dec)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        m = compute_epoch_metrics(
            threshold=threshold,
            scores_train=scores_train,
            scores_val=scores_val,
            scores_test=scores_test,
            interpretable_weight=0.90,
            splits=splits,
        )

        ep_label = f"Epoch {ep:02d}/{end_epoch:02d}"
        tr_metric = f"MSE={mean_loss:.5f}"
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
                model_type="LSTM_Autoencoder",
                best_epoch=ep,
                seq_len=seq_len,
                hidden_dim=H,
                W_enc=W_enc.copy(),
                U_enc=U_enc.copy(),
                b_enc=b_enc.copy(),
                W_ctx=W_ctx.copy(),
                b_ctx=b_ctx.copy(),
                W_dec=W_dec.copy(),
                b_dec=b_dec.copy(),
                loss_history=np.array(loss_history),
                score_threshold=threshold,
                composite_score=best_score,
                **splits["common_meta"],
            )

        print_epoch_row(ep_label, tr_metric, m, is_best)
        epoch_history.append({
            "epoch": ep,
            "train_mse_loss": round(mean_loss, 5),
            "val_roc_auc": round(m["val_roc_auc"], 4),
            "val_f1": m["val_bm"]["f1_score"],
            "test_roc_auc": round(m["test_roc_auc"], 4),
            "test_f1": m["test_bm"]["f1_score"],
            "composite_score": round(m["composite_score"], 2),
            "is_best": is_best,
        })

    return finalize_and_save_best(
        model_key="lstm_autoencoder",
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
