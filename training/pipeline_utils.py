import csv
import json
import math
from pathlib import Path
import numpy as np

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


def load_splits_From_disk() -> dict:
    ts_train, tag_names, X_train_raw = load_split_csv(SPLITS_DIR / "train_70.csv")
    ts_val, _, X_val_raw = load_split_csv(SPLITS_DIR / "val_15.csv")
    ts_test, _, X_test_raw = load_split_csv(SPLITS_DIR / "test_15.csv")

    n_train = len(X_train_raw)
    n_val = len(X_val_raw)
    n_test = len(X_test_raw)

    train_medians = np.nanmedian(X_train_raw, axis=0)
    X_all_raw = np.vstack([X_train_raw, X_val_raw, X_test_raw])
    X_all_imp = X_all_raw.copy()
    nan_rows, nan_cols = np.where(np.isnan(X_all_imp))
    X_all_imp[nan_rows, nan_cols] = train_medians[nan_cols]

    X_sel, sel_names, _ = engineer_features(X_all_imp, tag_names)
    X_train_sel = X_sel[:n_train]

    scaler_mean = np.mean(X_train_sel, axis=0)
    scaler_std = np.std(X_train_sel, axis=0)
    scaler_std = np.where(scaler_std < 1e-6, 1.0, scaler_std)

    X_scaled = (X_sel - scaler_mean) / scaler_std
    physics_mask_all = build_physics_audit_mask(X_all_raw, X_sel, tag_names, sel_names, n_train)

    return {
        "ts_train": ts_train,
        "ts_val": ts_val,
        "ts_test": ts_test,
        "ts_all": ts_train + ts_val + ts_test,
        "sel_names": sel_names,
        "X_train_scaled": X_scaled[:n_train],
        "X_val_scaled": X_scaled[n_train : n_train + n_val],
        "X_test_scaled": X_scaled[n_train + n_val :],
        "y_train_phys": physics_mask_all[:n_train],
        "y_val_phys": physics_mask_all[n_train : n_train + n_val],
        "y_test_phys": physics_mask_all[n_train + n_val :],
        "common_meta": {
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
            "train_medians": train_medians,
            "selected_features": np.array(sel_names),
        },
    }


def compute_loss_cost_rmse(per_sample_losses: np.ndarray) -> dict[str, float]:
    """
    Computes:
      - loss_median: typical single-sample loss L(x_t, x_hat_t)
      - cost_mse: Cost function J(theta) = (1/N) * sum L(x_t, x_hat_t)
      - rmse: Root Mean Squared Error = sqrt(J(theta))
    """
    cost_mse = float(np.mean(per_sample_losses))
    loss_median = float(np.median(per_sample_losses))
    rmse = math.sqrt(max(cost_mse, 0.0))
    return {
        "loss_single_sample_median": round(loss_median, 5),
        "cost_function_mse": round(cost_mse, 5),
        "rmse": round(rmse, 4),
    }


def compute_epoch_metrics(
    threshold: float,
    scores_train: np.ndarray,
    scores_val: np.ndarray,
    scores_test: np.ndarray,
    interpretable_weight: float,
    splits: dict,
    loss_train: np.ndarray | None = None,
    loss_val: np.ndarray | None = None,
    loss_test: np.ndarray | None = None,
) -> dict:
    raw_train = scores_train > threshold
    raw_val = scores_val > threshold
    raw_test = scores_test > threshold

    clean_val, events_val = consolidate_events(raw_val, splits["ts_val"])
    clean_test, events_test = consolidate_events(raw_test, splits["ts_test"])

    val_roc_auc = roc_auc_score_np(splits["y_val_phys"], scores_val)
    test_roc_auc = roc_auc_score_np(splits["y_test_phys"], scores_test)

    val_bm = binary_metrics_np(splits["y_val_phys"], clean_val)
    test_bm = binary_metrics_np(splits["y_test_phys"], clean_test)

    base_far = float(np.mean(raw_train) * 100.0)
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
        + 0.10 * interpretable_weight
        + 0.10 * stability_score
    )

    l_tr = compute_loss_cost_rmse(loss_train if loss_train is not None else scores_train)
    l_va = compute_loss_cost_rmse(loss_val if loss_val is not None else scores_val)
    l_te = compute_loss_cost_rmse(loss_test if loss_test is not None else scores_test)

    return {
        "threshold": threshold,
        "base_far": base_far,
        "train_loss_metrics": l_tr,
        "val_loss_metrics": l_va,
        "test_loss_metrics": l_te,
        "val_roc_auc": val_roc_auc,
        "val_bm": val_bm,
        "val_events": events_val,
        "clean_val": clean_val,
        "test_roc_auc": test_roc_auc,
        "test_bm": test_bm,
        "test_events": events_test,
        "raw_test": raw_test,
        "clean_test": clean_test,
        "snr_test": snr_test,
        "composite_score": composite_score,
    }


def print_banner(model_title: str, family: str, split_info: dict, resume_msg: str | None = None) -> None:
    line = "=" * 124
    print(f"\n{line}")
    print(f"  MODEL TRAINING & EVALUATION: {model_title}")
    print(f"  Architecture Family        : {family}")
    print(
        f"  Dataset Splits (70/15/15)  : Train={len(split_info['X_train_scaled']):,} rows | "
        f"Val={len(split_info['X_val_scaled']):,} rows | Test={len(split_info['X_test_scaled']):,} rows "
        f"({len(split_info['sel_names'])} selected features)"
    )
    if resume_msg:
        print(f"  Checkpoint Resume State    : {resume_msg}")
    else:
        print("  Checkpoint Resume State    : No prior checkpoint found — starting fresh from Epoch 01")
    print(line)
    print(
        f"  {'Epoch / Stage':<24} | {'Train Metric':<14} | {'Cost (MSE)':>10} | {'Test RMSE':>9} | "
        f"{'Val F1':>7} | {'Test AUC':>8} | {'Test F1':>7} | {'Score':>6} | {'Status'}"
    )
    print("-" * 124)


def print_epoch_row(
    epoch_label: str,
    train_metric_str: str,
    m: dict,
    is_best: bool,
    status_override: str | None = None,
) -> None:
    if status_override is not None:
        badge = status_override
    else:
        badge = "★ NEW BEST (Checkpoint Saved)" if is_best else ""
    cost_tr = m["train_loss_metrics"]["cost_function_mse"]
    rmse_te = m["test_loss_metrics"]["rmse"]
    print(
        f"  {epoch_label:<24} | {train_metric_str:<14} | "
        f"{cost_tr:>10.5f} | {rmse_te:>9.4f} | "
        f"{m['val_bm']['f1_score']:>7.4f} | "
        f"{m['test_roc_auc']:>8.4f} | {m['test_bm']['f1_score']:>7.4f} | "
        f"{m['composite_score']:>6.2f} | {badge}"
    )


def finalize_and_save_best(
    model_key: str,
    model_name: str,
    family: str,
    model_dir: Path,
    checkpoint_path: Path,
    best_epoch_label: str,
    best_train_metric: str,
    best_eval: dict,
    best_scores_test: np.ndarray,
    infer_ms_test: float,
    epoch_history: list[dict],
    splits: dict,
    extra_metadata: dict | None = None,
) -> dict:
    clean_val = best_eval["clean_val"]
    clean_test = best_eval["clean_test"]
    raw_test = best_eval["raw_test"]
    events_val = best_eval["val_events"]
    events_test = best_eval["test_events"]

    metrics = {
        "model_key": model_key,
        "model_name": model_name,
        "family": family,
        "best_epoch_or_stage": best_epoch_label,
        "best_train_metric": best_train_metric,
        "checkpoint_file": str(checkpoint_path.relative_to(ROOT_DIR)),
        "threshold_99pct": round(float(best_eval["threshold"]), 4),
        "training_split_70pct": {
            "rows": len(splits["X_train_scaled"]),
            "false_alarm_rate_pct": round(best_eval["base_far"], 2),
            **best_eval["train_loss_metrics"],
        },
        "validation_split_15pct": {
            "rows": len(splits["X_val_scaled"]),
            **best_eval["val_loss_metrics"],
            "roc_auc": round(best_eval["val_roc_auc"], 4),
            "f1_score": best_eval["val_bm"]["f1_score"],
            "precision": best_eval["val_bm"]["precision"],
            "recall": best_eval["val_bm"]["recall"],
            "accuracy": best_eval["val_bm"]["accuracy"],
            "flagged_pct_clean": round(float(np.mean(clean_val) * 100.0), 2),
            "candidate_fault_events": len(events_val),
        },
        "testing_split_15pct": {
            "rows": len(splits["X_test_scaled"]),
            **best_eval["test_loss_metrics"],
            "roc_auc": round(best_eval["test_roc_auc"], 4),
            "f1_score": best_eval["test_bm"]["f1_score"],
            "precision": best_eval["test_bm"]["precision"],
            "recall": best_eval["test_bm"]["recall"],
            "accuracy": best_eval["test_bm"]["accuracy"],
            "signal_to_noise_ratio": round(best_eval["snr_test"], 2),
            "flagged_pct_clean": round(float(np.mean(clean_test) * 100.0), 2),
            "candidate_fault_events": len(events_test),
            "inference_time_ms": round(infer_ms_test, 2),
        },
        "composite_project_score": round(best_eval["composite_score"], 2),
        "epoch_history": epoch_history,
    }
    if extra_metadata:
        metrics.update(extra_metadata)

    scores_json_path = model_dir / f"{model_key}_scores.json"
    scores_json_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    test_scores_csv_path = model_dir / f"{model_key}_test_scores.csv"
    with test_scores_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "anomaly_score", "raw_alarm_flag", "cleaned_event_flag"])
        for ts, sc, rf, cf in zip(splits["ts_test"], best_scores_test, raw_test, clean_test):
            writer.writerow([ts, f"{sc:.6f}", int(rf), int(cf)])

    events_csv_path = model_dir / f"{model_key}_candidate_events.csv"
    for ev in events_test:
        s, e = ev["start_idx"], ev["end_idx"] + 1
        ev["peak_anomaly_score"] = round(float(np.max(best_scores_test[s:e])), 4)
        ev["mean_anomaly_score"] = round(float(np.mean(best_scores_test[s:e])), 4)
    with events_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "event_id",
            "start_idx",
            "end_idx",
            "start_timestamp",
            "end_timestamp",
            "duration_min",
            "peak_anomaly_score",
            "mean_anomaly_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events_test)

    line = "=" * 124
    print("-" * 124)
    print(f"  FINAL & BEST RESULT SUMMARY — {model_name}")
    print("-" * 124)
    print(f"  Best Epoch / Stage Achieved : {best_epoch_label} ({best_train_metric})")
    print(f"  Overall Composite Score     : {metrics['composite_project_score']:.2f} / 100")
    print(
        f"  Loss & Cost Function (MSE)  : Train Loss(Med)={metrics['training_split_70pct']['loss_single_sample_median']:.5f} | "
        f"Train Cost(MSE)={metrics['training_split_70pct']['cost_function_mse']:.5f} | "
        f"Val Cost(MSE)={metrics['validation_split_15pct']['cost_function_mse']:.5f} | "
        f"Test Cost(MSE)={metrics['testing_split_15pct']['cost_function_mse']:.5f}"
    )
    print(
        f"  RMSE Across Splits          : Train RMSE = {metrics['training_split_70pct']['rmse']:.4f} | "
        f"Validation RMSE = {metrics['validation_split_15pct']['rmse']:.4f} | "
        f"Testing RMSE = {metrics['testing_split_15pct']['rmse']:.4f}"
    )
    print(
        f"  Validation (15%) Metrics    : ROC-AUC = {metrics['validation_split_15pct']['roc_auc']:.4f} | "
        f"F1 = {metrics['validation_split_15pct']['f1_score']:.4f} | "
        f"Precision = {metrics['validation_split_15pct']['precision']:.4f} | "
        f"Recall = {metrics['validation_split_15pct']['recall']:.4f} | "
        f"RMSE = {metrics['validation_split_15pct']['rmse']:.4f}"
    )
    print(
        f"  Testing (15%) Metrics       : ROC-AUC = {metrics['testing_split_15pct']['roc_auc']:.4f} | "
        f"F1 = {metrics['testing_split_15pct']['f1_score']:.4f} | "
        f"Precision = {metrics['testing_split_15pct']['precision']:.4f} | "
        f"Recall = {metrics['testing_split_15pct']['recall']:.4f} | "
        f"RMSE = {metrics['testing_split_15pct']['rmse']:.4f} | "
        f"Accuracy = {metrics['testing_split_15pct']['accuracy']*100:.2f}%"
    )
    print(f"  Auto-Saved Best Checkpoint  : {metrics['checkpoint_file']}")
    print(f"  Auto-Saved Model Scores     : {scores_json_path.relative_to(ROOT_DIR)}")
    print(f"{line}\n")

    return metrics
