"""Loads the Part 3 data.

Put the downloads into a data/ directory next to this file, then run:

    python load_data.py

Needs pandas (pip install pandas).
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"

# Nine predictive values. 
ATTRIBUTES = [
    "gender", "age", "education", "household_income", "industry",
    "employment_status", "employer_size", "home_ownership", "parental_status",
]


def load_users(split):
    return pd.read_json(DATA / f"{split}.jsonl.gz", lines=True, dtype=False)


def load_labels(split):
    if split == "test":
        raise ValueError("test has no labels")
    return pd.read_csv(DATA / f"{split}_labels.csv", dtype=str,
                       keep_default_na=False)


if __name__ == "__main__":
    for split in ("train", "dev", "test"):
        users = load_users(split)
        print(f"{split}: {len(users)} users")
        print(users["client_id"])
        if split != "test":
            print(load_labels(split)[["client_id", *ATTRIBUTES]])
        print()
