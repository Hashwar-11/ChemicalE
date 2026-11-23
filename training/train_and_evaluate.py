import csv
import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = Path(__file__).resolve().parent / "splits"
MODELS_DIR = Path(__file__).resolve().parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_split_csv(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        tag_names = header[1:]
        timestamps: list[str] = []
        rows: list[list[float]] = []
        for row in reader:
            timestamps.append(row[0])
            rows.append([float(x) if x != "" else np.nan for x in row[1:]])
    return timestamps, tag_names, np.array(rows, dtype=np.float64)


def rolling_mean_1d(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    counts = np.arange(1, window, dtype=np.float64)
    out[: window - 1] = cumsum[1:window] / counts
    return out


def rolling_std_1d(arr: np.ndarray, window: int) -> np.ndarray:
    mean1 = rolling_mean_1d(arr, window)
    mean2 = rolling_mean_1d(arr * arr, window)
    var = np.maximum(0.0, mean2 - mean1 * mean1)
    out = np.sqrt(var)
    out[0] = 0.0
    return out


def engineer_features(
    X_imp: np.ndarray,
    tag_names: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Milestone 5 (18 physics-based features) + Milestone 6 (Feature selection: 40 core TK-101 features).
    """
    idx = {name: i for i, name in enumerate(tag_names)}

    def col(name: str) -> np.ndarray:
        return X_imp[:, idx[name]]

    err_tic_101 = col("TIC-101.PV") - col("TIC-101.SP")
    err_lic_101 = col("LIC-101.PV") - col("LIC-101.SP")
    err_fic_201 = col("FIC-201.PV") - col("FIC-201.SP")

    res_duty = col("CALC-DUTY") - col("CALC-DUTY-PROC")
    ua_rolling_diff = col("CALC-UA") - rolling_mean_1d(col("CALC-UA"), 1440)

    valve_disc_tv101 = col("TV-101.POS") - col("FIC-201.OP")
    valve_disc_lv101 = col("LV-101.POS") - col("LIC-101.OP")

    diff_ti_201_202 = col("TI-201") - col("TI-202")
    diff_tank_in_out = col("TIC-101.PV") - col("TI-100")
    diff_core_exit_temp = col("TIC-101.PV") - col("TI-102")

    roll_std_tags = [
        "TIC-101.PV",
        "TIC-101.OP",
        "FIC-201.PV",
        "FIC-201.OP",
        "LIC-101.PV",
        "LIC-101.OP",
        "PI-201",
        "FI-101",
    ]
    roll_std_cols = [rolling_std_1d(col(t), 15) for t in roll_std_tags]
    roll_std_names = [f"{t}_ROLL_STD" for t in roll_std_tags]

    eng_cols = [
        err_tic_101,
        err_lic_101,
        err_fic_201,
        res_duty,
        ua_rolling_diff,
        valve_disc_tv101,
        valve_disc_lv101,
        diff_ti_201_202,
        diff_tank_in_out,
        diff_core_exit_temp,
        *roll_std_cols,
    ]
    eng_names = [
        "ERR_TIC-101",
        "ERR_LIC-101",
        "ERR_FIC-201",
        "RES_DUTY",
        "UA_ROLLING_DIFF",
        "VALVE_DISC_TV101",
        "VALVE_DISC_LV101",
        "DIFF_TI_201_202",
        "DIFF_TANK_IN_OUT",
        "DIFF_CORE_EXIT_TEMP",
        *roll_std_names,
    ]

    X_all = np.column_stack([X_imp, *eng_cols])
    all_names = tag_names + eng_names

    dropped_suffixes = (".MODE",)
    dropped_prefixes = ("CW-", "N2-", "SD-", "AIR-")
    dropped_exact = {
        "CALC-DUTY-PROC",
        "TI-100",
        "TIC-101.SP",
        "LIC-101.SP",
        "TI-901",
    }

    selected_indices: list[int] = []
    selected_names: list[str] = []

    for i, name in enumerate(all_names):
        if name.endswith(dropped_suffixes) or name.startswith(dropped_prefixes) or name in dropped_exact:
            continue
        selected_indices.append(i)
        selected_names.append(name)

    return X_all[:, selected_indices], selected_names, all_names


def consolidate_events(flags: np.ndarray, timestamps: list[str], min_spike: int = 2, merge_gap: int = 15) -> tuple[np.ndarray, list[dict]]:
    f = flags.astype(bool).copy()
    n = len(f)

    i = 0
    while i < n:
        if f[i]:
            j = i
            while j < n and f[j]:
                j += 1
            if (j - i) <= min_spike:
                f[i:j] = False
            i = j
        else:
            i += 1

    i = 0
    while i < n:
        if not f[i]:
            j = i
            while j < n and not f[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= merge_gap:
                f[i:j] = True
            i = j
        else:
            i += 1

    events: list[dict] = []
    i = 0
    event_id = 1
    while i < n:
        if f[i]:
            j = i
            while j < n and f[j]:
                j += 1
            events.append({
                "event_id": event_id,
                "start_idx": int(i),
                "end_idx": int(j - 1),
                "start_timestamp": timestamps[i],
                "end_timestamp": timestamps[j - 1],
                "duration_min": int(j - i),
            })
            event_id += 1
            i = j
        else:
            i += 1

    return f, events


def build_physics_audit_mask(
    X_all_raw: np.ndarray,
    X_eng: np.ndarray,
    tag_names: list[str],
    sel_names: list[str],
    train_end: int,
) -> np.ndarray:
    sidx = {name: i for i, name in enumerate(sel_names)}
    key_criteria = [
        ("CALC-UA", 3.0),
        ("UA_ROLLING_DIFF", 3.0),
        ("PI-201", 3.0),
        ("TI-201", 3.0),
        ("CALC-APPROACH", 3.0),
        ("ERR_TIC-101", 3.0),
        ("ERR_LIC-101", 3.0),
        ("ERR_FIC-201", 3.0),
        ("VALVE_DISC_TV101", 3.0),
        ("VALVE_DISC_LV101", 3.0),
        ("RES_DUTY", 3.0),
        ("CALC-EBAL", 3.0),
        ("CALC-MBAL", 3.0),
        ("DIFF_TI_201_202", 3.0),
        ("AI-101", 3.0),
        ("TIC-101.OP_ROLL_STD", 3.2),
        ("FIC-201.PV_ROLL_STD", 3.2),
        ("PI-201_ROLL_STD", 3.2),
    ]

    mask = np.zeros(len(X_eng), dtype=bool)
    for feat_name, z_thresh in key_criteria:
        col = X_eng[:, sidx[feat_name]]
        base = col[:train_end]
        med = np.median(base)
        mad = np.median(np.abs(base - med)) * 1.4826
        std = max(float(np.std(base)), float(mad), 1e-6)
        z = np.abs(col - med) / std
        mask |= (z > z_thresh)

    tk101_raw_idx = [
        i for i, t in enumerate(tag_names)
        if not t.startswith(("CW-", "N2-", "SD-", "AIR-"))
    ]
    comms_outage = np.isnan(X_all_raw[:, tk101_raw_idx]).any(axis=1)
    mask |= comms_outage

    f = mask.copy()
    n = len(f)
    i = 0
    while i < n:
        if f[i]:
            j = i
            while j < n and f[j]:
                j += 1
            if (j - i) <= 2:
                f[i:j] = False
            i = j
        else:
            i += 1
    return f


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sum_pos_ranks = np.sum(ranks[y_true])
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_t = y_true.astype(bool)
    y_p = y_pred.astype(bool)
    tp = int(np.sum(y_t & y_p))
    fp = int(np.sum((~y_t) & y_p))
    fn = int(np.sum(y_t & (~y_p)))
    tn = int(np.sum((~y_t) & (~y_p)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / len(y_t)
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


# =====================================================================
# 1. PCA Multivariate Monitor (Hotelling's T^2 + Q-statistic)
# =====================================================================
class PCAMonitorModel:
    def __init__(self, variance_threshold: float = 0.95, alpha_quantile: float = 0.99):
        self.variance_threshold = variance_threshold
        self.alpha_quantile = alpha_quantile
        self.components_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.n_components_: int = 0
        self.retained_variance_: float = 0.0
        self.t2_ucl_: float = 1.0
        self.q_ucl_: float = 1.0
        self.score_threshold_: float = 1.0

    def fit(self, X_train: np.ndarray, X_baseline: np.ndarray) -> "PCAMonitorModel":
        cov = np.cov(X_train, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 1e-8)
        eigvecs = eigvecs[:, order]

        cumvar = np.cumsum(eigvals) / np.sum(eigvals)
        k = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
        k = min(k, X_train.shape[1])

        self.n_components_ = k
        self.retained_variance_ = float(cumvar[k - 1])
        self.components_ = eigvecs[:, :k]
        self.eigenvalues_ = eigvals[:k]

        t2_base, q_base = self._raw_t2_q(X_baseline)
        self.t2_ucl_ = float(np.quantile(t2_base, self.alpha_quantile))
        self.q_ucl_ = float(np.quantile(q_base, self.alpha_quantile))

        combined_base = np.maximum(t2_base / self.t2_ucl_, q_base / self.q_ucl_)
        self.score_threshold_ = float(np.quantile(combined_base, self.alpha_quantile))
        return self

    def _raw_t2_q(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = X @ self.components_
        t2 = np.sum((scores ** 2) / self.eigenvalues_, axis=1)
        X_recon = scores @ self.components_.T
        q = np.sum((X - X_recon) ** 2, axis=1)
        return t2, q

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        t2, q = self._raw_t2_q(X)
        return np.maximum(t2 / self.t2_ucl_, q / self.q_ucl_)

    def feature_contributions(self, X_subset: np.ndarray) -> np.ndarray:
        scores = X_subset @ self.components_
        X_recon = scores @ self.components_.T
        q_contrib = (X_subset - X_recon) ** 2
        t2_weights = (self.components_ / np.sqrt(self.eigenvalues_)) @ self.components_.T
        t2_contrib = (X_subset @ t2_weights) ** 2
        return np.mean(q_contrib / self.q_ucl_ + t2_contrib / self.t2_ucl_, axis=0)

    def save_checkpoint(self, path: Path, common_meta: dict) -> None:
        np.savez_compressed(
            path,
            model_type="PCA_T2_Q",
            components=self.components_,
            eigenvalues=self.eigenvalues_,
            n_components=self.n_components_,
            retained_variance=self.retained_variance_,
            t2_ucl=self.t2_ucl_,
            q_ucl=self.q_ucl_,
            score_threshold=self.score_threshold_,
            **common_meta,
        )


# =====================================================================
# 2. LSTM Autoencoder (Temporal Sequence Reconstruction Network)
# =====================================================================
class LSTMAutoencoderModel:
    """
    Many-to-Many Recurrent LSTM Encoder-Decoder trained via Adam optimizer
    on sliding windows of length `seq_len` (10 minutes) to reconstruct
    multivariate temporal trajectories X_{t-L+1:t}.
    """

    def __init__(
        self,
        seq_len: int = 10,
        hidden_dim: int = 20,
        epochs: int = 12,
        batch_size: int = 256,
        lr: float = 0.008,
        alpha_quantile: float = 0.99,
        random_state: int = 42,
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.alpha_quantile = alpha_quantile
        self.random_state = random_state

        # LSTM Encoder weights: combined gates [f, i, g, o] of size 4*H
        self.W_enc: np.ndarray | None = None  # (D, 4H)
        self.U_enc: np.ndarray | None = None  # (H, 4H)
        self.b_enc: np.ndarray | None = None  # (4H,)
        # Temporal context projection & Decoder readout
        self.W_ctx: np.ndarray | None = None  # (2H, H)
        self.b_ctx: np.ndarray | None = None  # (H,)
        self.W_dec: np.ndarray | None = None  # (H, D)
        self.b_dec: np.ndarray | None = None  # (D,)
        self.loss_history_: list[float] = []
        self.score_threshold_: float = 1.0

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

    def _make_windows(self, X: np.ndarray, stride: int = 1) -> np.ndarray:
        # Shape: (N_win, L, D) using stride tricks
        n, d = X.shape
        L = self.seq_len
        # Pad front with first row repeated L-1 times so output aligns 1-to-1 with every timestamp
        X_pad = np.vstack([np.repeat(X[:1], L - 1, axis=0), X])
        s0, s1 = X_pad.strides
        n_win = (len(X_pad) - L) // stride + 1
        windows = np.lib.stride_tricks.as_strided(
            X_pad,
            shape=(n_win, L, d),
            strides=(s0 * stride, s0, s1),
        )
        return windows

    def _forward_lstm_batch(self, W_seq: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        W_seq: (B, L, D)
        Runs LSTM encoder across L steps, aggregates [h_L, mean(h_{1:L})],
        and decodes current state reconstruction x_hat_t.
        """
        B, L, D = W_seq.shape
        H = self.hidden_dim
        h = np.zeros((B, H), dtype=np.float64)
        c = np.zeros((B, H), dtype=np.float64)

        h_sum = np.zeros((B, H), dtype=np.float64)
        gates_cache = []

        for t in range(L):
            xt = W_seq[:, t, :]
            z = xt @ self.W_enc + h @ self.U_enc + self.b_enc  # (B, 4H)
            f = self._sigmoid(z[:, :H])
            i = self._sigmoid(z[:, H : 2 * H])
            g = np.tanh(np.clip(z[:, 2 * H : 3 * H], -15.0, 15.0))
            o = self._sigmoid(z[:, 3 * H :])

            c_next = f * c + i * g
            tanh_c = np.tanh(np.clip(c_next, -15.0, 15.0))
            h_next = o * tanh_c

            gates_cache.append((xt, h, c, f, i, g, o, c_next, tanh_c))
            h = h_next
            c = c_next
            h_sum += h

        h_mean = h_sum / L
        ctx_in = np.hstack([h, h_mean])  # (B, 2H)
        latent = np.tanh(ctx_in @ self.W_ctx + self.b_ctx)  # (B, H)
        x_hat = latent @ self.W_dec + self.b_dec  # (B, D)

        cache = {
            "gates": gates_cache,
            "ctx_in": ctx_in,
            "latent": latent,
            "L": L,
        }
        return x_hat, cache

    def fit(self, X_train: np.ndarray, X_baseline: np.ndarray) -> "LSTMAutoencoderModel":
        rng = np.random.default_rng(self.random_state)
        _, D = X_train.shape
        H = self.hidden_dim

        # Xavier initialization
        scale_enc = math.sqrt(2.0 / (D + H))
        self.W_enc = rng.normal(0.0, scale_enc, size=(D, 4 * H))
        self.U_enc = rng.normal(0.0, math.sqrt(1.0 / H), size=(H, 4 * H))
        self.b_enc = np.zeros(4 * H, dtype=np.float64)
        self.b_enc[:H] = 1.0  # Forget gate bias = 1.0 for stable temporal memory

        self.W_ctx = rng.normal(0.0, math.sqrt(2.0 / (3 * H)), size=(2 * H, H))
        self.b_ctx = np.zeros(H, dtype=np.float64)
        self.W_dec = rng.normal(0.0, math.sqrt(2.0 / (H + D)), size=(H, D))
        self.b_dec = np.zeros(D, dtype=np.float64)

        # Build strided training sequences (stride=3 -> ~30,240 sequences for fast mini-batch BPTT)
        train_wins = self._make_windows(X_train, stride=3)
        n_wins = len(train_wins)

        # Adam state for [W_enc, U_enc, b_enc, W_ctx, b_ctx, W_dec, b_dec]
        params = [self.W_enc, self.U_enc, self.b_enc, self.W_ctx, self.b_ctx, self.W_dec, self.b_dec]
        m_adam = [np.zeros_like(p) for p in params]
        v_adam = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        self.loss_history_ = []
        for _ in range(self.epochs):
            perm = rng.permutation(n_wins)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_wins, self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                W_b = train_wins[batch_idx]  # (B, L, D)
                B = len(W_b)
                target = W_b[:, -1, :]  # reconstruct current state x_t from sequence context

                x_hat, cache = self._forward_lstm_batch(W_b)
                err = x_hat - target
                loss = float(np.mean(err ** 2))
                epoch_loss += loss
                n_batches += 1

                # Backpropagation through decoder -> context -> LSTM step L & temporal mean
                d_xhat = (2.0 / (B * D)) * err  # (B, D)
                latent = cache["latent"]
                ctx_in = cache["ctx_in"]

                dW_dec = latent.T @ d_xhat
                db_dec = np.sum(d_xhat, axis=0)

                d_latent = (d_xhat @ self.W_dec.T) * (1.0 - latent ** 2)  # (B, H)
                dW_ctx = ctx_in.T @ d_latent
                db_ctx = np.sum(d_latent, axis=0)

                d_ctx_in = d_latent @ self.W_ctx.T  # (B, 2H)
                dh_last = d_ctx_in[:, :H]
                dh_mean = d_ctx_in[:, H:] / self.seq_len

                dW_enc = np.zeros_like(self.W_enc)
                dU_enc = np.zeros_like(self.U_enc)
                db_enc = np.zeros_like(self.b_enc)

                dh_next = dh_last.copy()
                dc_next = np.zeros((B, H), dtype=np.float64)

                # Truncated BPTT across last 4 time steps of the window for speed & stability
                gates = cache["gates"]
                for t in range(self.seq_len - 1, max(-1, self.seq_len - 5), -1):
                    xt, h_prev, c_prev, f, i, g, o, c_t, tanh_c = gates[t]
                    dh = dh_next + dh_mean
                    do = dh * tanh_c * o * (1.0 - o)
                    dc = dh * o * (1.0 - tanh_c ** 2) + dc_next
                    df = dc * c_prev * f * (1.0 - f)
                    di = dc * g * i * (1.0 - i)
                    dg = dc * i * (1.0 - g ** 2)

                    dz = np.hstack([df, di, dg, do])  # (B, 4H)
                    dW_enc += xt.T @ dz
                    dU_enc += h_prev.T @ dz
                    db_enc += np.sum(dz, axis=0)

                    dh_next = dz @ self.U_enc.T
                    dc_next = dc * f

                grads = [dW_enc, dU_enc, db_enc, dW_ctx, db_ctx, dW_dec, db_dec]
                step += 1
                for idx_p, (p, g_p) in enumerate(zip(params, grads)):
                    g_clip = np.clip(g_p, -1.0, 1.0)
                    m_adam[idx_p] = beta1 * m_adam[idx_p] + (1.0 - beta1) * g_clip
                    v_adam[idx_p] = beta2 * v_adam[idx_p] + (1.0 - beta2) * (g_clip ** 2)
                    m_hat = m_adam[idx_p] / (1.0 - beta1 ** step)
                    v_hat = v_adam[idx_p] / (1.0 - beta2 ** step)
                    p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            self.loss_history_.append(round(epoch_loss / max(n_batches, 1), 5))

        base_scores = self.score_samples(X_baseline)
        self.score_threshold_ = float(np.quantile(base_scores, self.alpha_quantile))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        wins = self._make_windows(X, stride=1)
        n = len(wins)
        scores = np.empty(n, dtype=np.float64)
        bs = 2048
        for start in range(0, n, bs):
            W_b = wins[start : start + bs]
            x_hat, _ = self._forward_lstm_batch(W_b)
            target = W_b[:, -1, :]
            scores[start : start + bs] = np.mean((target - x_hat) ** 2, axis=1)
        return scores

    def save_checkpoint(self, path: Path, common_meta: dict) -> None:
        np.savez_compressed(
            path,
            model_type="LSTM_Autoencoder",
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            W_enc=self.W_enc,
            U_enc=self.U_enc,
            b_enc=self.b_enc,
            W_ctx=self.W_ctx,
            b_ctx=self.b_ctx,
            W_dec=self.W_dec,
            b_dec=self.b_dec,
            loss_history=np.array(self.loss_history_),
            score_threshold=self.score_threshold_,
            **common_meta,
        )


# =====================================================================
# 3. Local Outlier Factor (LOF - Density-Based k-NN Novelty Detector)
# =====================================================================
class LocalOutlierFactorModel:
    """
    Breunig et al. (2000) Local Outlier Factor in Novelty Detection mode.
    Uses a representative reference pool of M=3,024 normal training states
    (sampled uniformly every 30 minutes across the 70% training timeline)
    with k=20 nearest neighbors.
    """

    def __init__(self, n_neighbors: int = 20, ref_size: int = 3024, alpha_quantile: float = 0.99):
        self.n_neighbors = n_neighbors
        self.ref_size = ref_size
        self.alpha_quantile = alpha_quantile
        self.X_ref_: np.ndarray | None = None
        self.ref_sq_norms_: np.ndarray | None = None
        self.k_dist_ref_: np.ndarray | None = None
        self.lrd_ref_: np.ndarray | None = None
        self.score_threshold_: float = 1.0

    def fit(self, X_train: np.ndarray, X_baseline: np.ndarray) -> "LocalOutlierFactorModel":
        stride = max(1, len(X_train) // self.ref_size)
        self.X_ref_ = X_train[::stride][: self.ref_size].copy()
        m = len(self.X_ref_)
        self.ref_sq_norms_ = np.sum(self.X_ref_ ** 2, axis=1)

        # Pairwise Euclidean distance matrix among reference points (M x M)
        d2 = np.maximum(
            0.0,
            self.ref_sq_norms_[:, None] + self.ref_sq_norms_[None, :] - 2.0 * (self.X_ref_ @ self.X_ref_.T),
        )
        np.fill_diagonal(d2, np.inf)
        dists = np.sqrt(d2)

        k = self.n_neighbors
        knn_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(m)[:, None]
        knn_dists = dists[row_idx, knn_idx]
        self.k_dist_ref_ = np.max(knn_dists, axis=1)

        # Reachability distance of reference points
        reach_dists = np.maximum(knn_dists, self.k_dist_ref_[knn_idx])
        self.lrd_ref_ = 1.0 / (np.mean(reach_dists, axis=1) + 1e-8)

        base_scores = self.score_samples(X_baseline)
        self.score_threshold_ = float(np.quantile(base_scores, self.alpha_quantile))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        scores = np.empty(n, dtype=np.float64)
        k = self.n_neighbors
        bs = 2048
        for start in range(0, n, bs):
            Xb = X[start : start + bs]
            xb_norms = np.sum(Xb ** 2, axis=1)
            d2 = np.maximum(
                0.0,
                xb_norms[:, None] + self.ref_sq_norms_[None, :] - 2.0 * (Xb @ self.X_ref_.T),
            )
            dists = np.sqrt(d2)
            knn_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(len(Xb))[:, None]
            knn_dists = dists[row_idx, knn_idx]

            reach_dists = np.maximum(knn_dists, self.k_dist_ref_[knn_idx])
            lrd_x = 1.0 / (np.mean(reach_dists, axis=1) + 1e-8)
            lof_x = np.mean(self.lrd_ref_[knn_idx], axis=1) / lrd_x
            scores[start : start + bs] = lof_x
        return scores

    def save_checkpoint(self, path: Path, common_meta: dict) -> None:
        np.savez_compressed(
            path,
            model_type="Local_Outlier_Factor",
            n_neighbors=self.n_neighbors,
            X_ref=self.X_ref_,
            k_dist_ref=self.k_dist_ref_,
            lrd_ref=self.lrd_ref_,
            score_threshold=self.score_threshold_,
            **common_meta,
        )


# =====================================================================
# 4. Isolation Forest (Tree Ensemble)
# =====================================================================
class IsolationForestModel:
    def __init__(
        self,
        n_estimators: int = 150,
        max_samples: int = 256,
        alpha_quantile: float = 0.99,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.alpha_quantile = alpha_quantile
        self.random_state = random_state
        self.max_depth = int(math.ceil(math.log2(max_samples)))
        self.trees_: list[dict] = []
        self.c_norm_: float = self._c_factor(max_samples)
        self.score_threshold_: float = 0.5

    @staticmethod
    def _c_factor(n: int) -> float:
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        h = math.log(n - 1.0) + 0.5772156649
        return 2.0 * h - (2.0 * (n - 1.0) / n)

    def _build_tree(self, X_sub: np.ndarray, depth: int, rng: np.random.Generator) -> dict:
        n_samples, n_features = X_sub.shape
        if depth >= self.max_depth or n_samples <= 1:
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
            "left": self._build_tree(X_sub[left_mask], depth + 1, rng),
            "right": self._build_tree(X_sub[~left_mask], depth + 1, rng),
        }

    def _eval_tree(self, node: dict, X: np.ndarray, idx: np.ndarray, out_lengths: np.ndarray) -> None:
        if len(idx) == 0:
            return
        if node["leaf"]:
            out_lengths[idx] = node["depth"] + self._c_factor(node["size"])
            return
        col = X[idx, node["feat"]]
        left_mask = col < node["split"]
        self._eval_tree(node["left"], X, idx[left_mask], out_lengths)
        self._eval_tree(node["right"], X, idx[~left_mask], out_lengths)

    def fit(self, X_train: np.ndarray, X_baseline: np.ndarray) -> "IsolationForestModel":
        rng = np.random.default_rng(self.random_state)
        n = len(X_train)
        self.trees_ = []
        for _ in range(self.n_estimators):
            sample_idx = rng.choice(n, size=min(self.max_samples, n), replace=False)
            tree = self._build_tree(X_train[sample_idx], 0, rng)
            self.trees_.append(tree)

        base_scores = self.score_samples(X_baseline)
        self.score_threshold_ = float(np.quantile(base_scores, self.alpha_quantile))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        total_lengths = np.zeros(n, dtype=np.float64)
        all_idx = np.arange(n, dtype=np.int64)
        tree_lengths = np.empty(n, dtype=np.float64)
        for tree in self.trees_:
            self._eval_tree(tree, X, all_idx, tree_lengths)
            total_lengths += tree_lengths
        mean_lengths = total_lengths / self.n_estimators
        return np.power(2.0, -mean_lengths / self.c_norm_)

    def save_checkpoint(self, path: Path, common_meta: dict) -> None:
        trees_json = json.dumps(self.trees_)
        np.savez_compressed(
            path,
            model_type="Isolation_Forest",
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            c_norm=self.c_norm_,
            trees_json=np.array(trees_json),
            score_threshold=self.score_threshold_,
            **common_meta,
        )


# =====================================================================
# 5. Regularized Mahalanobis / Elliptic Envelope
# =====================================================================
class MahalanobisEnvelopeModel:
    def __init__(self, reg_lambda: float = 1e-2, alpha_quantile: float = 0.99):
        self.reg_lambda = reg_lambda
        self.alpha_quantile = alpha_quantile
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.score_threshold_: float = 1.0

    def fit(self, X_train: np.ndarray, X_baseline: np.ndarray) -> "MahalanobisEnvelopeModel":
        self.mean_ = np.mean(X_baseline, axis=0)
        cov = np.cov(X_baseline, rowvar=False) + self.reg_lambda * np.eye(X_baseline.shape[1])
        self.precision_ = np.linalg.inv(cov)
        base_scores = self.score_samples(X_baseline)
        self.score_threshold_ = float(np.quantile(base_scores, self.alpha_quantile))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        diff = X - self.mean_
        dist_sq = np.sum((diff @ self.precision_) * diff, axis=1)
        return np.sqrt(np.maximum(dist_sq, 0.0))

    def save_checkpoint(self, path: Path, common_meta: dict) -> None:
        np.savez_compressed(
            path,
            model_type="Mahalanobis_Envelope",
            reg_lambda=self.reg_lambda,
            mean=self.mean_,
            precision=self.precision_,
            score_threshold=self.score_threshold_,
            **common_meta,
        )


# =====================================================================
# 6. K-Means Operating-Mode Clustering Anomaly Detector
# =====================================================================
class KMeansAnomalyModel:
    def __init__(self, n_clusters: int = 5, max_iter: int = 40, alpha_quantile: float = 0.99, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.alpha_quantile = alpha_quantile
        self.random_state = random_state
        self.centroids_: np.ndarray | None = None
        self.score_threshold_: float = 1.0

    def fit(self, X_train: np.ndarray, X_baseline: np.ndarray) -> "KMeansAnomalyModel":
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X_train), size=self.n_clusters, replace=False)
        centroids = X_train[idx].copy()

        for _ in range(self.max_iter):
            dists = np.sum((X_train[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = centroids.copy()
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = np.mean(X_train[mask], axis=0)
            if np.allclose(new_centroids, centroids, atol=1e-4):
                centroids = new_centroids
                break
            centroids = new_centroids

        self.centroids_ = centroids
        base_scores = self.score_samples(X_baseline)
        self.score_threshold_ = float(np.quantile(base_scores, self.alpha_quantile))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        dists = np.sum((X[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2)
        return np.sqrt(np.min(dists, axis=1))

    def save_checkpoint(self, path: Path, common_meta: dict) -> None:
        np.savez_compressed(
            path,
            model_type="KMeans_Clustering",
            n_clusters=self.n_clusters,
            centroids=self.centroids_,
            score_threshold=self.score_threshold_,
            **common_meta,
        )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading 70/15/15 splits...")
    ts_train, tag_names, X_train_raw = load_split_csv(SPLITS_DIR / "train_70.csv")
    ts_val, _, X_val_raw = load_split_csv(SPLITS_DIR / "val_15.csv")
    ts_test, _, X_test_raw = load_split_csv(SPLITS_DIR / "test_15.csv")

    n_train = len(X_train_raw)
    n_val = len(X_val_raw)
    n_test = len(X_test_raw)
    timestamps_all = ts_train + ts_val + ts_test

    # Step 1: Leakage-Free Imputation (fit median on Train 70% only)
    train_medians = np.nanmedian(X_train_raw, axis=0)
    X_all_raw = np.vstack([X_train_raw, X_val_raw, X_test_raw])
    X_all_imp = X_all_raw.copy()
    nan_rows, nan_cols = np.where(np.isnan(X_all_imp))
    X_all_imp[nan_rows, nan_cols] = train_medians[nan_cols]

    # Step 2: Feature Engineering (Milestone 5) & Feature Selection (Milestone 6)
    X_sel, sel_names, all_names = engineer_features(X_all_imp, tag_names)
    print(f"Total features after Milestone 5 engineering: {len(all_names)}")
    print(f"Selected features after Milestone 6 filtering: {len(sel_names)}")

    X_train_sel = X_sel[:n_train]
    X_val_sel = X_sel[n_train : n_train + n_val]
    X_test_sel = X_sel[n_train + n_val :]

    n_baseline = n_train
    X_base_sel = X_train_sel

    # Step 3: Leakage-Free Standardization (fit on 70% training set only)
    scaler_mean = np.mean(X_train_sel, axis=0)
    scaler_std = np.std(X_train_sel, axis=0)
    scaler_std = np.where(scaler_std < 1e-6, 1.0, scaler_std)

    X_base_scaled = (X_base_sel - scaler_mean) / scaler_std
    X_train_scaled = (X_train_sel - scaler_mean) / scaler_std
    X_val_scaled = (X_val_sel - scaler_mean) / scaler_std
    X_test_scaled = (X_test_sel - scaler_mean) / scaler_std

    common_checkpoint_meta = {
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "train_medians": train_medians,
        "selected_features": np.array(sel_names),
    }

    # Step 4: Construct Physics-Verified Benchmark Audit Mask
    physics_mask_all = build_physics_audit_mask(X_all_raw, X_sel, tag_names, sel_names, n_train)
    y_val_phys = physics_mask_all[n_train : n_train + n_val]
    y_test_phys = physics_mask_all[n_train + n_val :]

    # Step 5: Train, Evaluate & Save Checkpoints for All 6 Candidate ML Models
    models = {
        "PCA_T2_Q": {
            "display_name": "PCA Multivariate Monitor (Hotelling T² + Q-Stat)",
            "instance": PCAMonitorModel(variance_threshold=0.95, alpha_quantile=0.99),
            "family": "Subspace Projection & Residual SPC",
            "checkpoint_file": "pca_t2_q/pca_t2_q_checkpoint.npz",
            "interpretable": 1.00,
        },
        "LSTM_Autoencoder": {
            "display_name": "LSTM Autoencoder (SeqLen=10, Hidden=20)",
            "instance": LSTMAutoencoderModel(seq_len=10, hidden_dim=20, epochs=12, batch_size=256, lr=0.008, alpha_quantile=0.99, random_state=42),
            "family": "Deep Temporal Sequence Reconstruction",
            "checkpoint_file": "lstm_autoencoder/lstm_autoencoder_checkpoint.npz",
            "interpretable": 0.90,
        },
        "Local_Outlier_Factor": {
            "display_name": "Local Outlier Factor (LOF, k=20 Novelty)",
            "instance": LocalOutlierFactorModel(n_neighbors=20, ref_size=3024, alpha_quantile=0.99),
            "family": "Local Density k-NN Anomaly Detection",
            "checkpoint_file": "local_outlier_factor/lof_checkpoint.npz",
            "interpretable": 0.80,
        },
        "Isolation_Forest": {
            "display_name": "Isolation Forest (150 Trees)",
            "instance": IsolationForestModel(n_estimators=150, max_samples=256, alpha_quantile=0.99, random_state=42),
            "family": "Tree-Based Random Partitioning",
            "checkpoint_file": "isolation_forest/isolation_forest_checkpoint.npz",
            "interpretable": 0.78,
        },
    }

    comparison_results: list[dict] = []
    all_model_scores: dict[str, np.ndarray] = {}
    checkpoints_manifest: list[dict] = []

    for key, info in models.items():
        print(f"Training {info['display_name']}...")
        model = info["instance"]
        t0 = time.perf_counter()
        model.fit(X_train_scaled, X_base_scaled)
        fit_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        scores_test = model.score_samples(X_test_scaled)
        test_infer_ms = (time.perf_counter() - t1) * 1000.0

        scores_base = model.score_samples(X_base_scaled)
        scores_train = scores_base
        scores_val = model.score_samples(X_val_scaled)
        scores_all = np.concatenate([scores_train, scores_val, scores_test])
        all_model_scores[key] = scores_all

        # Save model checkpoint strictly inside its own model folder
        ckpt_path = MODELS_DIR / info["checkpoint_file"]
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(ckpt_path, common_checkpoint_meta)

        thresh = model.score_threshold_

        raw_base = scores_base > thresh
        raw_train = scores_train > thresh
        raw_val = scores_val > thresh
        raw_test = scores_test > thresh
        raw_all = scores_all > thresh

        clean_val, events_val = consolidate_events(raw_val, ts_val)
        clean_test, events_test = consolidate_events(raw_test, ts_test)
        clean_all, events_all = consolidate_events(raw_all, timestamps_all)

        val_roc_auc = roc_auc_score_np(y_val_phys, scores_val)
        test_roc_auc = roc_auc_score_np(y_test_phys, scores_test)

        val_bm = binary_metrics_np(y_val_phys, clean_val)
        test_bm = binary_metrics_np(y_test_phys, clean_test)

        base_far = float(np.mean(raw_base) * 100.0)
        far_calibration_score = max(0.0, 1.0 - abs(base_far - 1.0) / 5.0)

        noise_ratio_test = 1.0 - (float(np.sum(clean_test)) / max(int(np.sum(raw_test)), 1))
        stability_score = float(np.clip(1.0 - abs(noise_ratio_test - 0.15), 0.0, 1.0))

        normal_mean = float(np.mean(scores_test[~clean_test])) if np.any(~clean_test) else 1.0
        anom_mean = float(np.mean(scores_test[clean_test])) if np.any(clean_test) else normal_mean
        snr_test = anom_mean / max(normal_mean, 1e-6)

        composite_score = 100.0 * (
            0.35 * test_roc_auc
            + 0.30 * test_bm["f1_score"]
            + 0.15 * far_calibration_score
            + 0.10 * info["interpretable"]
            + 0.10 * stability_score
        )

        res = {
            "model_key": key,
            "model_name": info["display_name"],
            "family": info["family"],
            "checkpoint_path": str(ckpt_path.relative_to(ROOT_DIR)),
            "threshold_99pct": round(thresh, 4),
            "baseline_far_pct": round(base_far, 2),
            "train_flag_pct": round(float(np.mean(raw_train) * 100.0), 2),
            "val_flag_pct_raw": round(float(np.mean(raw_val) * 100.0), 2),
            "val_flag_pct_clean": round(float(np.mean(clean_val) * 100.0), 2),
            "val_events": len(events_val),
            "val_roc_auc": round(val_roc_auc, 4),
            "val_f1": val_bm["f1_score"],
            "val_precision": val_bm["precision"],
            "val_recall": val_bm["recall"],
            "test_flag_pct_raw": round(float(np.mean(raw_test) * 100.0), 2),
            "test_flag_pct_clean": round(float(np.mean(clean_test) * 100.0), 2),
            "test_events": len(events_test),
            "test_roc_auc": round(test_roc_auc, 4),
            "test_f1": test_bm["f1_score"],
            "test_precision": test_bm["precision"],
            "test_recall": test_bm["recall"],
            "test_accuracy": test_bm["accuracy"],
            "test_snr": round(snr_test, 2),
            "full_90d_events": len(events_all),
            "fit_time_s": round(fit_time_s, 3),
            "test_infer_ms": round(test_infer_ms, 2),
            "composite_score": round(composite_score, 2),
        }
        comparison_results.append(res)

    comparison_results.sort(key=lambda r: r["composite_score"], reverse=True)
    best_model_res = comparison_results[0]
    best_model_key = best_model_res["model_key"]

    for rank, r in enumerate(comparison_results, start=1):
        checkpoints_manifest.append({
            "rank": rank,
            "is_best": rank == 1,
            "model_key": r["model_key"],
            "model_name": r["model_name"],
            "checkpoint_file": r["checkpoint_path"],
            "composite_score": r["composite_score"],
            "test_roc_auc": r["test_roc_auc"],
            "test_f1": r["test_f1"],
            "val_roc_auc": r["val_roc_auc"],
            "val_f1": r["val_f1"],
            "threshold_99pct": r["threshold_99pct"],
        })

    (MODELS_DIR / "checkpoints_manifest.json").write_text(
        json.dumps(
            {
                "best_checkpoint": best_model_res["checkpoint_path"],
                "best_model_key": best_model_key,
                "best_model_name": best_model_res["model_name"],
                "best_composite_score": best_model_res["composite_score"],
                "checkpoints": checkpoints_manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pca_model: PCAMonitorModel = models["PCA_T2_Q"]["instance"]  # type: ignore
    lstm_model: LSTMAutoencoderModel = models["LSTM_Autoencoder"]["instance"]  # type: ignore
    top_contrib = pca_model.feature_contributions(
        X_test_scaled[all_model_scores["PCA_T2_Q"][n_train + n_val :] > pca_model.score_threshold_]
    )
    top_feat_idx = np.argsort(top_contrib)[::-1][:10]
    top_features_test = [
        {"feature": sel_names[i], "mean_contribution": round(float(top_contrib[i]), 3)}
        for i in top_feat_idx
    ]

    csv_fields = list(comparison_results[0].keys())
    with (RESULTS_DIR / "model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(comparison_results)

    best_scores_test = all_model_scores[best_model_key][n_train + n_val :]
    best_thresh = models[best_model_key]["instance"].score_threshold_
    _, best_test_events = consolidate_events(best_scores_test > best_thresh, ts_test)
    for ev in best_test_events:
        s, e = ev["start_idx"], ev["end_idx"] + 1
        ev["peak_anomaly_score"] = round(float(np.max(best_scores_test[s:e])), 3)
        contrib_ev = pca_model.feature_contributions(X_test_scaled[s:e])
        top3 = np.argsort(contrib_ev)[::-1][:3]
        ev["top_contributing_features"] = ", ".join(f"{sel_names[idx]}" for idx in top3)

    if best_test_events:
        with (RESULTS_DIR / "test_candidate_fault_events.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(best_test_events[0].keys()))
            writer.writeheader()
            writer.writerows(best_test_events)

    summary = {
        "split_sizes": {"train_70": n_train, "val_15": n_val, "test_15": n_test, "clean_baseline": n_baseline},
        "feature_counts": {
            "raw_tags": len(tag_names),
            "engineered_total": len(all_names),
            "selected_features": len(sel_names),
            "pca_components_95pct": pca_model.n_components_,
            "pca_retained_variance": round(pca_model.retained_variance_, 4),
            "pca_t2_ucl_99pct": round(pca_model.t2_ucl_, 2),
            "pca_q_ucl_99pct": round(pca_model.q_ucl_, 2),
            "lstm_loss_history": lstm_model.loss_history_,
        },
        "best_model": best_model_res,
        "model_ranking": comparison_results,
        "top_10_test_anomaly_features": top_features_test,
        "test_candidate_events_count": len(best_test_events),
    }
    (RESULTS_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Generate comparison bar chart across all 6 models
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=150)
    names = [r["model_name"].split(" (")[0] for r in comparison_results]
    x = np.arange(len(names))
    w = 0.25

    axes[0].bar(x - w, [r["test_roc_auc"] for r in comparison_results], width=w, label="Test ROC-AUC", color="#1d4ed8")
    axes[0].bar(x, [r["test_f1"] for r in comparison_results], width=w, label="Test F1-Score", color="#059669")
    axes[0].bar(x + w, [r["val_f1"] for r in comparison_results], width=w, label="Val F1-Score", color="#d97706")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Score (0.0 – 1.0)")
    axes[0].set_title("Validation & Test Metrics Across All 6 Candidate Models")
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    colors = ["#64748b", "#94a3b8", "#f59e0b", "#d97706", "#2563eb", "#059669"]
    bars = axes[1].barh(
        names[::-1],
        [r["composite_score"] for r in comparison_results][::-1],
        color=colors[-len(names) :],
    )
    axes[1].set_xlim(0, 108)
    axes[1].set_xlabel("Composite Engineering Score (0 – 100)")
    axes[1].set_title("Overall Project Score & Checkpoint Ranking")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)
    for bar in bars:
        val = bar.get_width()
        axes[1].text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison_chart.png")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
