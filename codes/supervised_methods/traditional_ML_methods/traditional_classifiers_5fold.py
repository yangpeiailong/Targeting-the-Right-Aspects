import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import jieba
import os
from datetime import datetime
from pathlib import Path

# ===================== Core Configuration (Adapt to your split files) =====================
DATA_NAME = "哪吒之魔童闹海的影评"  # Dataset name (keep original for path consistency)
DATA_ROOT_PATH = str(Path(__file__).parent.parent.parent.parent.absolute()) + "/data/"
# List of pre-split 5-fold files (use your specified path format)
FOLD_FILE_LIST = [f"{DATA_NAME}_summary_version (deepseek-cot)_train_val_test_5_fold_{num}.xlsx" for num in range(1, 6)]
# Result save path (unified naming with deep learning version)
RESULT_FILENAME = f"multi_label_tr_model_comparison_{DATA_NAME}_summary_version (deepseek-cot)_5fold.xlsx"
FULL_SAVE_PATH = os.path.join(DATA_ROOT_PATH, "results_for_supervised_methods_full", RESULT_FILENAME)

# Create save directory if not exists
os.makedirs(os.path.dirname(FULL_SAVE_PATH), exist_ok=True)


# ===================== Metric Calculation Function (Align with deep learning logic) =====================
def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics for multi-label classification (consistent with deep learning version)

    Args:
        y_true: Ground truth labels (2D array)
        y_pred: Predicted labels (2D array)

    Returns:
        dict: Comprehensive metrics including overall and label-wise scores
    """
    # Flatten for multi-label scenario
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate overall metrics
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision_macro = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    recall_macro = recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    f1_macro = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    precision_micro = precision_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
    recall_micro = recall_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
    f1_micro = f1_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)

    # Calculate label-wise metrics
    label_wise_metrics = []
    for label_idx in range(y_true.shape[1]):
        y_true_label = y_true[:, label_idx]
        y_pred_label = y_pred[:, label_idx]
        label_wise_metrics.append({
            'label': label_idx,
            'precision': precision_score(y_true_label, y_pred_label, average='macro', zero_division=0),
            'recall': recall_score(y_true_label, y_pred_label, average='macro', zero_division=0),
            'f1': f1_score(y_true_label, y_pred_label, average='macro', zero_division=0)
        })

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'label_wise_metrics': label_wise_metrics
    }


# ===================== Single Fold Training Function (Adapt to pre-split files) =====================
def train_single_fold(fold_num, file_path):
    """
    Train model on a single fold (read pre-split fold file with train/val/test sheets)

    Args:
        fold_num: Fold number (1-5)
        file_path: Path to the fold file

    Returns:
        tuple: (fold_metrics_results, fold_label_wise_metrics)
               - fold_metrics_results: Overall metrics of all models in this fold
               - fold_label_wise_metrics: Label-wise metrics of all models in this fold
    """
    print(f"\n{'=' * 60}")
    print(f"Start processing Fold {fold_num}, File path: {file_path}")
    print(f"{'=' * 60}")

    # Read train/val/test sheets from fold file (no header, header=None)
    excel_file = pd.ExcelFile(file_path)
    train_df = excel_file.parse('train', header=None)
    val_df = excel_file.parse('val', header=None)
    test_df = excel_file.parse('test', header=None)

    # Extract text and labels (1st column: text, remaining columns: labels) + Jieba tokenization
    # Training set
    X_train = train_df.iloc[:, 0].astype(str).tolist()  # Ensure string type
    X_train = [' '.join(jieba.lcut(i)) for i in X_train]
    y_train = train_df.iloc[:, 1:].values

    # Validation set (for observation only; ML models use train set for training and test set for evaluation)
    X_val = val_df.iloc[:, 0].astype(str).tolist()
    X_val = [' '.join(jieba.lcut(i)) for i in X_val]
    y_val = val_df.iloc[:, 1:].values

    # Test set
    X_test = test_df.iloc[:, 0].astype(str).tolist()
    X_test = [' '.join(jieba.lcut(i)) for i in X_test]
    y_test = test_df.iloc[:, 1:].values

    # TF-IDF vectorization (fit only on training set to avoid data leakage)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Define multi-label classifiers (traditional ML models)
    classifiers = {
        "Logistic Regression": MultiOutputClassifier(LogisticRegression(max_iter=1000)),
        "Support Vector Machine": MultiOutputClassifier(SVC()),
        "Random Forest": MultiOutputClassifier(RandomForestClassifier())
    }

    fold_metrics_results = []  # Store metrics of all models in this fold
    fold_label_wise_metrics = []  # Store label-wise metrics of all models in this fold

    # Train each model
    for model_name, clf in classifiers.items():
        print(f"\n--- Fold {fold_num} - Training {model_name} ---")

        # Train model (using training set)
        clf.fit(X_train_vec, y_train)

        # Evaluate on test set (consistent with deep learning version: final evaluation on test set)
        y_test_pred = clf.predict(X_test_vec)
        metrics = calculate_metrics(y_test, y_test_pred)

        # Print results for current fold and model
        print(f"Fold {fold_num} - {model_name} Test Set Results:")
        print(
            f"Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['f1_macro']:.4f} | Micro F1: {metrics['f1_micro']:.4f}")

        # Save results (align with deep learning version's fields)
        fold_metrics_results.append({
            'fold_num': fold_num,
            'model_name': model_name,
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_macro': metrics['f1_macro'],
            'precision_micro': metrics['precision_micro'],
            'recall_micro': metrics['recall_micro'],
            'f1_micro': metrics['f1_micro']
        })

        # Collect label-wise metrics (align with deep learning version's fields)
        for label_metric in metrics['label_wise_metrics']:
            fold_label_wise_metrics.append({
                'fold_num': fold_num,
                'model_name': model_name,
                'label': label_metric['label'],
                'precision': label_metric['precision'],
                'recall': label_metric['recall'],
                'f1': label_metric['f1']
            })

    return fold_metrics_results, fold_label_wise_metrics


# ===================== 5-Fold Cross Validation Main Function (Core: Unified save format) =====================
def train_5_fold_cv():
    """Iterate over 5 pre-split fold files, complete training, and save results in deep learning format"""
    # Initialize result storage (consistent with deep learning version's structure)
    all_fold_metrics = []  # Overall metrics of all folds
    all_label_wise_metrics = []  # Label-wise metrics of all folds

    # Iterate over 5 fold files
    for fold_idx, file_name in enumerate(FOLD_FILE_LIST, 1):
        file_path = os.path.join(DATA_ROOT_PATH, "reviews_after_split", file_name)

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Warning: Fold {fold_idx} file not found - {file_path}, skipped")
            continue

        # Train current fold
        fold_result, fold_label_metric = train_single_fold(fold_idx, file_path)

        # Collect results of current fold
        all_fold_metrics.extend(fold_result)
        all_label_wise_metrics.extend(fold_label_metric)

    # ===================== Data Format Conversion (Align with deep learning version) =====================
    if not all_fold_metrics:
        print("Error: No valid fold results, terminated")
        return

    # 1. DataFrame for 5-fold detailed results (consistent sheet name/fields with deep learning version)
    df_all_folds = pd.DataFrame(all_fold_metrics)

    # 2. Model summary statistics (mean + std, consistent with deep learning version)
    agg_funcs = {
        'accuracy': ['mean', 'std'],
        'precision_macro': ['mean', 'std'],
        'recall_macro': ['mean', 'std'],
        'f1_macro': ['mean', 'std'],
        'precision_micro': ['mean', 'std'],
        'recall_micro': ['mean', 'std'],
        'f1_micro': ['mean', 'std']
    }
    df_summary = df_all_folds.groupby('model_name').agg(agg_funcs).round(4)
    # Rename columns (e.g., accuracy_mean, accuracy_std, consistent with deep learning version)
    df_summary.columns = ['_'.join(col).strip() for col in df_summary.columns.values]
    df_summary = df_summary.reset_index()

    # 3. DataFrame for label-wise metrics (consistent sheet name/fields with deep learning version)
    df_label_wise_metrics = pd.DataFrame(all_label_wise_metrics)

    # ===================== Save to single Excel file (3 sheets, fully aligned with deep learning version) =====================
    with pd.ExcelWriter(FULL_SAVE_PATH, engine='openpyxl') as writer:
        # All folds detailed results (sheet name consistent with deep learning version)
        df_all_folds.to_excel(writer, sheet_name='5-Fold Detailed Results', index=False)
        # Model summary statistics (mean + std for 5 folds, core comparison metrics)
        df_summary.to_excel(writer, sheet_name='Model Summary Statistics', index=False)
        # Label-wise detailed metrics (precision/recall/F1 for each label)
        df_label_wise_metrics.to_excel(writer, sheet_name='Label-wise Metrics', index=False)

    # ===================== Print results (for easy comparison) =====================
    print(f"\n{'=' * 80}")
    print("5-Fold Cross Validation Training Completed! Results saved in deep learning unified format:")
    print(f"Save Path: {FULL_SAVE_PATH}")
    print(f"{'=' * 80}")
    print("\n【Model Summary Statistics (5-Fold Mean + Std)】")
    print(df_summary)
    print(f"\nExcel contains 3 sheets:")
    print("1. 5-Fold Detailed Results: Raw metrics of each model in each fold")
    print("2. Model Summary Statistics: Mean + Std of each model across 5 folds (core comparison metrics)")
    print("3. Label-wise Metrics: Precision/Recall/F1 of each model for each label in each fold")


# ===================== Execute 5-Fold Cross Validation =====================
if __name__ == "__main__":
    train_5_fold_cv()