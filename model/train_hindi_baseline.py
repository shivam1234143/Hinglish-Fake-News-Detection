"""Train a leakage-aware Hinglish fake-news classifier.

Input: data/hinglish_news.csv
Required columns: title,text,label
Optional: source,topic,language,source_id,group_id,split
Labels: 0=FAKE, 1=REAL

The model combines word and character TF-IDF with a calibrated Linear SVM.
Character features help with Roman-Hindi spelling variation such as sarkar/sarkaar/sarcar.
"""
from pathlib import Path
import argparse, json, re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]


def clean(text):
    text = str(text or "").replace("\u200b", " ").replace("\ufeff", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text, flags=re.I)
    text = re.sub(r"@[A-Za-z0-9_]+", " USER ", text)
    text = re.sub(r"#([A-Za-z0-9_]+)", r" \1 ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def split_data(df, seed=42):
    if "split" in df.columns:
        s = df["split"].astype(str).str.lower()
        if (s == "test").any() and (s == "train").any():
            train = df[s == "train"].copy()
            test = df[s == "test"].copy()
            return train, test, "explicit split column"

    group_col = next((c for c in ("group_id", "event_id", "story_id") if c in df.columns and df[c].notna().all()), None)
    if group_col:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        tr, te = next(splitter.split(df, df["label"], groups=df[group_col]))
        return df.iloc[tr].copy(), df.iloc[te].copy(), f"group split by {group_col}"

    tr, te = train_test_split(df, test_size=0.20, random_state=seed, stratify=df["label"])
    return tr.copy(), te.copy(), "stratified random split"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "hinglish_news.csv"))
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    path = Path(args.data)
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = {"title", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=["title", "text", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["content"] = (df["title"].map(clean) + " " + df["text"].map(clean)).str.strip()
    df = df[df["content"].str.len() >= 40]
    df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)

    train, test, split_method = split_data(df, args.seed)
    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        raise SystemExit("Train/test must both contain FAKE and REAL examples.")

    word = TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=args.min_df,
                           max_df=0.995, sublinear_tf=True, max_features=180000,
                           token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=args.min_df,
                           sublinear_tf=True, max_features=220000)
    features = FeatureUnion([("word", word), ("char", char)])

    X_train = features.fit_transform(train["content"])
    X_test = features.transform(test["content"])
    clf = CalibratedClassifierCV(LinearSVC(C=1.5, class_weight="balanced", max_iter=10000), cv=3, method="sigmoid", n_jobs=-1)
    clf.fit(X_train, train["label"].astype(int))
    pred = clf.predict(X_test)

    acc = accuracy_score(test["label"], pred)
    p, r, f1, _ = precision_recall_fscore_support(test["label"], pred, average="binary", zero_division=0)
    macro_f1 = precision_recall_fscore_support(test["label"], pred, average="macro", zero_division=0)[2]
    print("\n=== SATYACHECK HINGLISH MODEL ===")
    print(f"Rows: {len(df):,} | Train: {len(train):,} | Test: {len(test):,}")
    print("Split:", split_method)
    print(f"Accuracy: {acc:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | Macro-F1: {macro_f1:.4f}")
    print("\nConfusion matrix [FAKE, REAL]:\n", confusion_matrix(test["label"], pred))
    print("\n", classification_report(test["label"], pred, target_names=["FAKE", "REAL"], digits=4))

    out = ROOT / "model"
    out.mkdir(exist_ok=True)
    joblib.dump(clf, out / "hinglish_fake_news_model.pkl", compress=3)
    joblib.dump(features, out / "hinglish_features.pkl", compress=3)
    metrics = {
        "model": "Calibrated LinearSVC + word/character TF-IDF",
        "features": "word 1-3 grams + char_wb 3-5 grams",
        "labels": {"0": "FAKE", "1": "REAL"},
        "decision_threshold": 0.60,
        "rows": int(len(df)), "train_rows": int(len(train)), "test_rows": int(len(test)),
        "split_method": split_method, "random_seed": args.seed,
        "accuracy": float(acc), "precision": float(p), "recall": float(r),
        "f1": float(f1), "macro_f1": float(macro_f1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nSaved model artifacts to", out)

if __name__ == "__main__":
    main()
