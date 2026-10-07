from pathlib import Path
import pandas as pd
import re

# ---------------------------------------------------------
# SatyaCheck - HinFakeNews Dataset Preparation
# ---------------------------------------------------------

INPUT_FILE = Path("data/raw/HinFakeNews-V1.xlsx")
OUTPUT_FILE = Path("data/hindi_news.csv")

SHEET_NAME = "Real&Fake Combined"


def clean_text(value):
    """Clean article text while preserving Hindi/Hinglish characters."""

    if pd.isna(value):
        return ""

    text = str(value)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():

    print("=" * 60)
    print("SATYACHECK - HINFAKENEWS DATASET PREPARATION")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    print(f"\nReading dataset:")
    print(INPUT_FILE)

    # Read only the required columns
    df = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        usecols=["URL", "TITLE", "CONTENT", "BOOL"]
    )

    print(f"\nOriginal rows: {len(df):,}")

    print("\nOriginal columns:")
    print(df.columns.tolist())

    # -----------------------------------------------------
    # Rename columns
    # -----------------------------------------------------

    df = df.rename(columns={
        "URL": "source_url",
        "TITLE": "title",
        "CONTENT": "text",
        "BOOL": "label"
    })

    # -----------------------------------------------------
    # Keep only required columns
    # -----------------------------------------------------

    df = df[
        ["title", "text", "label", "source_url"]
    ]

    # -----------------------------------------------------
    # Clean text
    # -----------------------------------------------------

    print("\nCleaning text...")

    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)
    df["source_url"] = df["source_url"].fillna("").astype(str).str.strip()

    # -----------------------------------------------------
    # Convert label to integer
    # -----------------------------------------------------

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    # Remove invalid labels
    df = df[df["label"].isin([0, 1])]

    df["label"] = df["label"].astype(int)

    # -----------------------------------------------------
    # Remove empty records
    # -----------------------------------------------------

    before = len(df)

    df = df[
        (df["title"].str.len() > 5) &
        (df["text"].str.len() > 50)
    ]

    removed_empty = before - len(df)

    print(f"Removed empty/very short records: {removed_empty:,}")

    # -----------------------------------------------------
    # Create combined text
    # -----------------------------------------------------

    df["text"] = (
        df["title"] + " " + df["text"]
    ).str.strip()

    # -----------------------------------------------------
    # Remove exact duplicate articles
    # -----------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["text", "label"]
    )

    duplicates_removed = before - len(df)

    print(f"Removed duplicate articles: {duplicates_removed:,}")

    # -----------------------------------------------------
    # Remove duplicate URLs
    # -----------------------------------------------------

    before = len(df)

    valid_url = df["source_url"].str.len() > 5

    df_url = df[valid_url].drop_duplicates(
        subset=["source_url"],
        keep="first"
    )

    df_no_url = df[~valid_url]

    df = pd.concat(
        [df_url, df_no_url],
        ignore_index=True
    )

    url_duplicates_removed = before - len(df)

    print(
        f"Removed duplicate URLs: "
        f"{url_duplicates_removed:,}"
    )

    # -----------------------------------------------------
    # Sort columns
    # -----------------------------------------------------

    df = df[
        ["title", "text", "label", "source_url"]
    ]

    # -----------------------------------------------------
    # Dataset statistics
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Final rows: {len(df):,}")

    print("\nLabel distribution:")

    counts = df["label"].value_counts().sort_index()

    fake_count = counts.get(0, 0)
    real_count = counts.get(1, 0)

    print(f"FAKE (0): {fake_count:,}")
    print(f"REAL (1): {real_count:,}")

    if len(df) > 0:

        print("\nClass percentages:")

        print(
            f"FAKE: {fake_count / len(df) * 100:.2f}%"
        )

        print(
            f"REAL: {real_count / len(df) * 100:.2f}%"
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSaving dataset...")

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(f"\nCreated:")
    print(OUTPUT_FILE)

    print("\nColumns:")

    print(df.columns.tolist())

    print("\nFirst record:")

    print(df.iloc[0][
        ["title", "label"]
    ].to_string())

    print("\nDataset is ready for validation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
OUTPUT_FILE = Path("data/hindi_news.csv")
