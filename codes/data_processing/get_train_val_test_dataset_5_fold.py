import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def split_5fold_train_val_test(raw_file_path, output_dir, dataname, random_seed=42):
    """
    Revised version: Implement 5 independent full-scale splits for multi-label text classification data.
    Each fold is split into train (60%)/validation (20%)/test (20%).
    Each fold contains 1000 samples in total, and the train/val/test subsheets in output files do NOT include headers.
    """
    # 1. Read raw data
    try:
        df = pd.read_excel(raw_file_path, header=0)
        print(f"Successfully read raw data: {len(df)} total records, {df.shape[1]} columns (1 text column + {df.shape[1] - 1} label columns)")
    except Exception as e:
        print(f"Failed to read raw file: {e}")
        return

    # 2. Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 3. Iterate 5 times for full-scale data splitting (adjust random seed for reproducible different splits)
    for fold_num in range(1, 6):
        # Adjust random seed (base seed + fold number to ensure unique & reproducible splits per fold)
        current_seed = random_seed + fold_num

        # 4. Step 1: Split into test set (20% of total) and train_val set (80% of total)
        train_val_df, test_df = train_test_split(
            df,
            test_size=0.2,  # 20% of total data (200 samples)
            random_state=current_seed
        )

        # 5. Step 2: Split train_val set into train (60% of total) and val (20% of total)
        # Note: train_val_df is 80% of total data, so 25% of it = 20% of total data for val set
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.25,  # 20% of total data (200 samples), remaining 60% for train (600 samples)
            random_state=current_seed
        )

        # 6. Define output file path for current fold
        output_file = os.path.join(
            output_dir,
            f"{dataname}_train_val_test_5_fold_{fold_num}.xlsx"
        )

        # 7. Write to xlsx file without headers
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            train_df.to_excel(writer, sheet_name="train", index=False, header=False)
            val_df.to_excel(writer, sheet_name="val", index=False, header=False)
            test_df.to_excel(writer, sheet_name="test", index=False, header=False)

        print(f"Fold {fold_num} data saved to: {output_file}")
        print(f"  - Train set: {len(train_df)} records | Val set: {len(val_df)} records | Test set: {len(test_df)} records (all without headers)")


# ===================== Usage Example =====================
if __name__ == "__main__":
    dataname = "哪吒之魔童闹海的影评"  # Keep Chinese for file naming (as per original path)
    DATA_ROOT_PATH = str(Path(__file__).parent.parent.parent.absolute())
    raw_file_path = os.path.join(DATA_ROOT_PATH, f"data/reviews_after_annotation/{dataname}_attributes_for_each_review2.xlsx")
    output_dir = os.path.join(DATA_ROOT_PATH, f"data/reviews_after_split")

    split_5fold_train_val_test(
        raw_file_path=raw_file_path,
        output_dir=output_dir,
        dataname=dataname,
        random_seed=42
    )