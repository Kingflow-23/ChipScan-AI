import csv
from pathlib import Path


def append_rows_to_csv(csv_path: Path, rows: list[dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())

        if write_header:
            writer.writeheader()

        writer.writerows(rows)
