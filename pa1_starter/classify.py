import sys
from pathlib import Path

import joblib
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
MODEL_PATH = Path(__file__).resolve().parent / "models.joblib"


def load_users(split):
    return pd.read_json(DATA / f"{split}.jsonl.gz", lines=True, dtype=False)


def load_labels(split):
    return pd.read_csv(DATA / f"{split}_labels.csv", dtype=str, keep_default_na=False)


def user_text(searches, ua):
    queries = " ".join(s["query"] for s in searches or [] if "query" in s)
    return queries + " " + (ua or "")


def extract_text(users):
    return users.apply(
        lambda row: user_text(row.get("searches"), row.get("latest_user_agent")), axis=1
    )


def make_pipeline():
    return Pipeline([
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


def train_serving_models():
    """Fit one pipeline per attribute on train + dev and save them for the SSP."""
    users = pd.concat([load_users("train"), load_users("dev")], ignore_index=True)
    labels = pd.concat([load_labels("train"), load_labels("dev")]).set_index("client_id")
    text = extract_text(users)

    # Every attribute sees the same text, so fit TF-IDF once and share it:
    # the saved file and the server's memory hold a single vocabulary.
    tfidf = make_pipeline().named_steps["tfidf"]
    X = tfidf.fit_transform(text)

    models = {}
    for attr in ATTRIBUTES:
        clf = make_pipeline().named_steps["clf"]
        clf.fit(X, labels.loc[users["client_id"], attr].values)
        models[attr] = Pipeline([("tfidf", tfidf), ("clf", clf)])
    joblib.dump(models, MODEL_PATH)
    return models


def load_serving_models():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_serving_models()


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

        pipe = make_pipeline()
        pipe.fit(train_text, y_train)
        dev_pred = pipe.predict(dev_text)
        score = f1_score(y_dev, dev_pred, average="macro", zero_division=0)
        print(f"{attr:20s}  macro-F1 = {score:.4f}")

        predictions[attr] = pipe.predict(test_text).tolist()

    out = pd.DataFrame(predictions)
    out.to_csv(DATA.parent / "predictions.csv", index=False)
    print("\nWrote predictions.csv")


if __name__ == "__main__":
    if "--serve" in sys.argv:
        train_serving_models()
        print(f"Wrote {MODEL_PATH}")
    else:
        main()
