import json
from pathlib import Path

from models.pca_t2_q.test_and_evaluate_pca_t2_q import run_evaluation as eval_pca
from models.lstm_autoencoder.test_and_evaluate_lstm_autoencoder import run_evaluation as eval_lstm
from models.local_outlier_factor.test_and_evaluate_lof import run_evaluation as eval_lof
from models.isolation_forest.test_and_evaluate_isolation_forest import run_evaluation as eval_iforest

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = [
        eval_pca(),
        eval_lstm(),
        eval_lof(),
        eval_iforest(),
    ]
    results.sort(key=lambda r: r["composite_project_score"], reverse=True)

    summary_path = RESULTS_DIR / "all_models_test_evaluation.json"
    summary_path.write_text(json.dumps({"ranked_models": results}, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "best_checkpoint": results[0]["checkpoint_file"],
        "best_model_key": results[0]["model_key"],
        "best_model_name": results[0]["model_name"],
        "best_composite_score": results[0]["composite_project_score"],
        "checkpoints": [
            {
                "rank": idx,
                "is_best": idx == 1,
                "model_key": r["model_key"],
                "model_name": r["model_name"],
                "best_epoch_or_stage": r["best_epoch_or_stage"],
                "checkpoint_file": r["checkpoint_file"],
                "composite_score": r["composite_project_score"],
                "train_cost_mse": r["training_split_70pct"]["cost_function_mse"],
                "train_rmse": r["training_split_70pct"]["rmse"],
                "val_rmse": r["validation_split_15pct"]["rmse"],
                "test_rmse": r["testing_split_15pct"]["rmse"],
                "test_roc_auc": r["testing_split_15pct"]["roc_auc"],
                "test_f1": r["testing_split_15pct"]["f1_score"],
                "val_roc_auc": r["validation_split_15pct"]["roc_auc"],
                "val_f1": r["validation_split_15pct"]["f1_score"],
                "threshold_99pct": r["threshold_99pct"],
            }
            for idx, r in enumerate(results, start=1)
        ],
    }
    (MODELS_DIR / "checkpoints_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    line = "=" * 124
    print(f"\n{line}")
    print("  OVERALL MODEL COMPARISON LEADERBOARD (LOSS, COST, RMSE, F1-SCORE & COMPOSITE SCORE)")
    print(line)
    print(
        f"  {'Rank':<5} | {'Model Name':<42} | {'Train Cost':>10} | {'Train RMSE':>10} | "
        f"{'Test RMSE':>9} | {'Test AUC':>8} | {'Test F1':>8} | {'Score':>6}"
    )
    print("-" * 124)
    for idx, r in enumerate(results, start=1):
        star = "★" if idx == 1 else " "
        print(
            f"  {star} #{idx:<2} | {r['model_name'][:42]:<42} | "
            f"{r['training_split_70pct']['cost_function_mse']:>10.5f} | "
            f"{r['training_split_70pct']['rmse']:>10.4f} | "
            f"{r['testing_split_15pct']['rmse']:>9.4f} | "
            f"{r['testing_split_15pct']['roc_auc']:>8.4f} | {r['testing_split_15pct']['f1_score']:>8.4f} | "
            f"{r['composite_project_score']:>6.2f}"
        )
    print(line)
    print(f"  ★ Winner Model      : {results[0]['model_name']} (Score: {results[0]['composite_project_score']:.2f}/100)")
    print(f"  ★ Winner Checkpoint : {results[0]['checkpoint_file']}")
    print(f"{line}\n")


if __name__ == "__main__":
    main()
