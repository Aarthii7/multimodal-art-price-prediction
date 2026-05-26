import os
import pandas as pd


# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "auction_final.csv")
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "auction_final_cleaned.csv")


def clean_price(price):
    if pd.isna(price):
        return None

    price = str(price)
    price = price.replace("USD", "")
    price = price.replace(".", "")  # remove thousand separator
    price = price.strip()

    try:
        return float(price)
    except:
        return None


def main():

    print("Loading dataset...")
    df = pd.read_csv(RAW_CSV_PATH)

    print("Original rows:", len(df))

    # =========================
    # CLEAN PRICE
    # =========================
    df["price"] = df["price"].apply(clean_price)

    # =========================
    # CLEAN YEAR
    # =========================
    df["yearCreation"] = pd.to_numeric(
        df["yearCreation"],
        errors="coerce"
    )

    # =========================
    # REMOVE INVALID PRICE/YEAR
    # =========================
    df = df.dropna(subset=["price", "yearCreation"])

    # =========================
    # REMOVE ZERO OR NEGATIVE PRICE
    # =========================
    df = df[df["price"] > 0]

    # =========================
    # CHECK IMAGE EXISTS
    # =========================
    valid_rows = []

    for _, row in df.iterrows():
        img_path = str(row["image_path"])

        # Case 1: Already absolute path
        if os.path.exists(img_path):
            row["image_path"] = img_path
            valid_rows.append(row)
            continue

        # Case 2: Filename only → join with image folder
        full_path = os.path.join(IMAGE_FOLDER, img_path)

        if os.path.exists(full_path):
            row["image_path"] = full_path
            valid_rows.append(row)

    df = pd.DataFrame(valid_rows)

    print("Rows after cleaning:", len(df))

    # =========================
    # FILL OPTIONAL COLUMNS
    # =========================
    optional_cols = [
        "artist", "signed", "condition",
        "period", "movement", "title"
    ]

    for col in optional_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # =========================
    # SAVE CLEAN FILE
    # =========================
    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved cleaned dataset to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()