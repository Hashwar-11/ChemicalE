import csv
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT_DIR / "historian_60s (1).csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "splits"


def count_missing(rows: list[list[str]]) -> tuple[int, int]:
    missing_cells = 0
    rows_with_missing = 0
    for row in rows:
        row_missing = sum(1 for cell in row[1:] if cell == "" or cell.lower() == "nan")
        if row_missing > 0:
            missing_cells += row_missing
            rows_with_missing += 1
    return missing_cells, rows_with_missing


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Sort chronologically by ISO-8601 UTC timestamp string (first column)
    rows.sort(key=lambda r: r[0])

    n_total = len(rows)
    n_train = int(round(n_total * 0.70))
    n_val = int(round(n_total * 0.15))
    n_test = n_total - n_train - n_val

    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]

    train_path = OUTPUT_DIR / "train_70.csv"
    val_path = OUTPUT_DIR / "val_15.csv"
    test_path = OUTPUT_DIR / "test_15.csv"
    meta_path = OUTPUT_DIR / "split_metadata.json"

    write_csv(train_path, header, train_rows)
    write_csv(val_path, header, val_rows)
    write_csv(test_path, header, test_rows)

    train_mc, train_rm = count_missing(train_rows)
    val_mc, val_rm = count_missing(val_rows)
    test_mc, test_rm = count_missing(test_rows)

    metadata = {
        "source_file": INPUT_CSV.name,
        "split_strategy": "chronological_70_15_15",
        "total_rows": n_total,
        "columns": len(header),
        "splits": {
            "train": {
                "file": str(train_path.relative_to(ROOT_DIR)),
                "ratio": 0.70,
                "rows": len(train_rows),
                "start_timestamp": train_rows[0][0],
                "end_timestamp": train_rows[-1][0],
                "missing_cells": train_mc,
                "rows_with_missing": train_rm,
            },
            "validation": {
                "file": str(val_path.relative_to(ROOT_DIR)),
                "ratio": 0.15,
                "rows": len(val_rows),
                "start_timestamp": val_rows[0][0],
                "end_timestamp": val_rows[-1][0],
                "missing_cells": val_mc,
                "rows_with_missing": val_rm,
            },
            "test": {
                "file": str(test_path.relative_to(ROOT_DIR)),
                "ratio": 0.15,
                "rows": len(test_rows),
                "start_timestamp": test_rows[0][0],
                "end_timestamp": test_rows[-1][0],
                "missing_cells": test_mc,
                "rows_with_missing": test_rm,
            },
        },
    }

    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

