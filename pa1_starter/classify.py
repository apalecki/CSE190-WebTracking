from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

ATTRIBUTES = [
    "gender", "age", "education", "household_income", "industry",
    "employment_status", "employer_size", "home_ownership", "parental_status",
]

DATA = Path(__file__).resolve().parent / "data"


def load_users(split):
    return pd.read_json(DATA / f"{split}.jsonl.gz", lines=True, dtype=False)


def load_labels(split):
    return pd.read_csv(DATA / f"{split}_labels.csv", dtype=str, keep_default_na=False)


def extract_text(users):
    def user_text(row):
        searches = row.get("searches") or []
        queries = " ".join(s["query"] for s in searches if "query" in s)
        ua = row.get("latest_user_agent") or ""
        return queries + " " + ua

    return users.apply(user_text, axis=1)


def main():
    train_users = load_users("train")
    train_labels = load_labels("train")
    dev_users = load_users("dev")
    dev_labels = load_labels("dev")
    test_users = load_users("test")

    train_labels = train_labels.set_index("client_id")
    dev_labels = dev_labels.set_index("client_id")

    train_text = extract_text(train_users)
    dev_text = extract_text(dev_users)
    test_text = extract_text(test_users)

    predictions = {"client_id": test_users["client_id"].tolist()}

    for attr in ATTRIBUTES:
        y_train = train_labels.loc[train_users["client_id"], attr].values
        y_dev = dev_labels.loc[dev_users["client_id"], attr].values

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                max_features=100000,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=5.0,
                solver="lbfgs",
                class_weight="balanced",
            )),
        ])

        pipe.fit(train_text, y_train)
        dev_pred = pipe.predict(dev_text)
        score = f1_score(y_dev, dev_pred, average="macro", zero_division=0)
        print(f"{attr:20s}  macro-F1 = {score:.4f}")

        predictions[attr] = pipe.predict(test_text).tolist()

    out = pd.DataFrame(predictions)
    out.to_csv(DATA.parent / "predictions.csv", index=False)
    print("\nWrote predictions.csv")


if __name__ == "__main__":
    main()
