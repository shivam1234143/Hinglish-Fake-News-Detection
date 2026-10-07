import os
import re
import pandas as pd


INPUT_FILE = "data/hinglish_corpus/PHINC.csv"
OUTPUT_FILE = "data/hinglish_corpus/phinc_clean.csv"


def clean_text(text):
    """Clean Hinglish text while preserving Roman Hindi and English."""

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove Twitter-style user mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove markdown-like formatting
    text = re.sub(r"[*_`~]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"PHINC dataset not found: {INPUT_FILE}"
        )

    print("Loading PHINC dataset...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Original rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # PHINC contains Sentence and English_Translation
    required_columns = ["Sentence", "English_Translation"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    # Keep only the Hinglish sentence and English translation
    df = df[required_columns].copy()

    df["Sentence"] = df["Sentence"].apply(clean_text)
    df["English_Translation"] = df["English_Translation"].apply(clean_text)

    # Remove empty sentences
    df = df[df["Sentence"].str.len() >= 5]

    # Remove duplicate Hinglish sentences
    df = df.drop_duplicates(
        subset=["Sentence"]
    ).reset_index(drop=True)

    # Rename columns to clearer names
    df = df.rename(
        columns={
            "Sentence": "hinglish_text",
            "English_Translation": "english_translation"
        }
    )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 50)
    print("PHINC PREPARATION COMPLETE")
    print("=" * 50)
    print(f"Final rows: {len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print()
    print("Sample:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()