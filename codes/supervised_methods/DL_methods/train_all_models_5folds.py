import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
from datetime import datetime
from config import MODEL_CONFIGS
from data_utils import load_data, get_dataloaders
from models import *
from pathlib import Path


# Assume MODEL_CONFIGS is defined elsewhere, containing all non-annotated model configurations
# MODEL_CONFIGS = {
#     'RandomAttLinear': {'class': 'RandomAttLinear', 'params': {}, 'save_path': '...'},
#     ...
# }

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics for multi-label classification"""
    # Flatten arrays (for multi-label scenarios)
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

    # Calculate metrics for each individual label
    label_metrics = []
    for label_idx in range(y_true.shape[1]):
        y_true_label = y_true[:, label_idx]
        y_pred_label = y_pred[:, label_idx]
        label_metrics.append({
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
        'label_metrics': label_metrics
    }


def train_all_models(data_path, reading_file, fold_num, epochs=20):
    """Train all non-annotated models (single fold) and return results"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []

    # Get all non-annotated models (assumed filtered in config)
    model_names = [name for name in MODEL_CONFIGS.keys()]
    print(f"\n[Fold {fold_num}] Models to be trained: {model_names}")

    for model_name in model_names:
        print(f"\n{'=' * 50}")
        print(f"[Fold {fold_num}] Start training model: {model_name}")
        print(f"{'=' * 50}")

        try:
            # Load model configuration
            config = MODEL_CONFIGS[model_name]
            is_bert = config['class'].startswith('Bert')
            model_type = "bert" if is_bert else "base"

            # Load data
            data = load_data(data_path, reading_file, model_type=model_type)
            dataloaders = get_dataloaders(data, batch_size=32, model_type=model_type)

            # Initialize model
            model_cls = globals()[config['class']]
            if is_bert:
                model_params = {**config['params'], **{
                    'model': config.get('model', 'D:/bert-base-chinese'),  # BERT model name/path
                    'maxlen': data['maxlen'],
                    'label_num': data['label_num'],
                    'label_class': data['label_class'],
                    'embed_size': config.get('embed_size', 768)  # BERT default hidden dimension
                }}
            else:
                model_params = {**config['params'], **{
                    'vocab': data['vocab'],
                    'word2idx': data['word2idx'],
                    'maxlen': data['maxlen'],
                    'label_num': data['label_num'],
                    'label_class': data['label_class'],
                    'embed_size': config.get('embed_size', 100)
                }}

            model = model_cls(**model_params).to(device)

            # Optimizer and loss function
            optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-4))
            loss_fn = nn.CrossEntropyLoss().to(device)

            # Training process
            best_val_f1 = 0.0
            best_epoch = 0

            for epoch in tqdm(range(epochs), desc=f"[Fold {fold_num}] Training {model_name}"):
                # Training phase
                model.train()
                total_loss = 0.0
                for batch in dataloaders['train']:
                    if model_type == "bert":
                        input_ids, attention_mask, token_type_ids, y = batch
                        x = (input_ids.to(device), attention_mask.to(device), token_type_ids.to(device))
                    else:
                        x, y = batch
                        x = x.to(device)

                    y = y.to(device)
                    optimizer.zero_grad()
                    pred = model(x)

                    # Reshape predictions and labels to match loss function
                    pred_reshaped = pred.reshape(-1, data['label_class'])
                    y_reshaped = y.argmax(dim=-1).reshape(-1)  # Convert from one-hot to indices

                    loss = loss_fn(pred_reshaped, y_reshaped)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                avg_train_loss = total_loss / len(dataloaders['train'])
                print(f"[Fold {fold_num}] Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

                # Validation phase
                model.eval()
                val_preds = []
                val_labels = []
                with torch.no_grad():
                    for batch in dataloaders['val']:
                        if model_type == "bert":
                            input_ids, attention_mask, token_type_ids, y = batch
                            x = (input_ids.to(device), attention_mask.to(device), token_type_ids.to(device))
                        else:
                            x, y = batch
                            x = x.to(device)

                        pred = model(x)
                        pred_argmax = torch.argmax(pred, dim=-1).cpu().numpy()
                        y_argmax = torch.argmax(y, dim=-1).cpu().numpy()

                        val_preds.append(pred_argmax)
                        val_labels.append(y_argmax)

                # Calculate validation metrics
                val_preds = np.concatenate(val_preds, axis=0)
                val_labels = np.concatenate(val_labels, axis=0)
                val_metrics = calculate_metrics(val_labels, val_preds)

                print(f"[Fold {fold_num}] Validation Metrics - Accuracy: {val_metrics['accuracy']:.4f}, "
                      f"Macro-F1: {val_metrics['f1_macro']:.4f}, "
                      f"Micro-F1: {val_metrics['f1_micro']:.4f}")

                # Save best model (based on macro-f1)
                if val_metrics['f1_macro'] > best_val_f1:
                    best_val_f1 = val_metrics['f1_macro']
                    best_epoch = epoch + 1
                    # Add fold suffix to model path to avoid overwriting
                    fold_save_path = config['save_path'].replace('.pth', f'_fold{fold_num}.pth')
                    torch.save(model.state_dict(), fold_save_path)
                    print(f"[Fold {fold_num}] Updated best model (Epoch {best_epoch}), Save Path: {fold_save_path}")

            # Testing phase
            fold_save_path = config['save_path'].replace('.pth', f'_fold{fold_num}.pth')
            model.load_state_dict(torch.load(fold_save_path))
            model.eval()
            test_preds = []
            test_labels = []
            with torch.no_grad():
                for batch in dataloaders['test']:
                    if model_type == "bert":
                        input_ids, attention_mask, token_type_ids, y = batch
                        x = (input_ids.to(device), attention_mask.to(device), token_type_ids.to(device))
                    else:
                        x, y = batch
                        x = x.to(device)

                    pred = model(x)
                    pred_argmax = torch.argmax(pred, dim=-1).cpu().numpy()
                    y_argmax = torch.argmax(y, dim=-1).cpu().numpy()

                    test_preds.append(pred_argmax)
                    test_labels.append(y_argmax)

            # Calculate test metrics
            test_preds = np.concatenate(test_preds, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)
            test_metrics = calculate_metrics(test_labels, test_preds)

            # Save results for current fold and model
            results.append({
                'fold_num': fold_num,
                'model_name': model_name,
                'best_epoch': best_epoch,
                'accuracy': test_metrics['accuracy'],
                'precision_macro': test_metrics['precision_macro'],
                'recall_macro': test_metrics['recall_macro'],
                'f1_macro': test_metrics['f1_macro'],
                'precision_micro': test_metrics['precision_micro'],
                'recall_micro': test_metrics['recall_micro'],
                'f1_micro': test_metrics['f1_micro'],
                'label_metrics': test_metrics['label_metrics']
            })

        except Exception as e:
            print(f"[Fold {fold_num}] Error training {model_name}: {str(e)}")
            continue

    return results


def train_all_models_5folds(data_path, reading_file_list, result_file, epochs=20):
    """
    Train all models with 5-fold cross-validation, summarize results and save
    :param data_path: Root path of dataset
    :param reading_file_list: List of 5-fold data files
    :param result_file: Filename for saving results
    :param epochs: Number of training epochs
    :return: Summarized results of all folds
    """
    # Initialize result storage
    all_fold_results = []
    all_label_metrics = []

    # Iterate over 5 folds
    for fold_idx, reading_file in enumerate(reading_file_list, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing Fold {fold_idx}, File: {reading_file}")
        print(f"{'=' * 60}")

        # Train all models for current fold
        fold_result = train_all_models(
            data_path=data_path,
            reading_file=reading_file,
            fold_num=fold_idx,
            epochs=epochs
        )

        # Collect current fold results
        all_fold_results.extend(fold_result)

        # Extract label-level metrics for current fold
        for res in fold_result:
            for label_metric in res['label_metrics']:
                all_label_metrics.append({
                    'fold_num': res['fold_num'],
                    'model_name': res['model_name'],
                    'label': label_metric['label'],
                    'precision': label_metric['precision'],
                    'recall': label_metric['recall'],
                    'f1': label_metric['f1']
                })

    # ========== Result Organization and Saving ==========
    # 1. Convert to DataFrame
    df_all_folds = pd.DataFrame(all_fold_results)
    df_label_metrics = pd.DataFrame(all_label_metrics)

    # Remove label_metrics column (saved separately)
    df_all_folds = df_all_folds.drop(columns=['label_metrics'])

    # 2. Calculate 5-fold statistical metrics (mean + std) for each model
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
    # Rename columns (e.g.: accuracy_mean, accuracy_std)
    df_summary.columns = ['_'.join(col).strip() for col in df_summary.columns.values]
    df_summary = df_summary.reset_index()

    # 3. Create save directory
    save_dir = os.path.join(data_path, 'results_for_supervised_methods_full')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    full_save_path = os.path.join(save_dir, result_file)

    # 4. Save to Excel (multiple sheets)
    with pd.ExcelWriter(full_save_path, engine='openpyxl') as writer:
        # Detailed results of all folds
        df_all_folds.to_excel(writer, sheet_name='5-Fold Detailed Results', index=False)
        # Model summary statistics (mean + std)
        df_summary.to_excel(writer, sheet_name='Model Summary Statistics', index=False)
        # Detailed metrics for each label
        df_label_metrics.to_excel(writer, sheet_name='Label-Level Detailed Metrics', index=False)

    print(f"\nAll 5-fold training completed! Results saved to: {full_save_path}")
    print("\nModel Summary Statistics:")
    print(df_summary)

    return {
        'all_fold_results': df_all_folds,
        'model_summary': df_summary,
        'label_metrics': df_label_metrics
    }


# Execute training
if __name__ == "__main__":
    # Configuration parameters
    DATA_PATH = str(Path(__file__).parent.parent.parent.parent.absolute()) + "/data/reviews_after_split"  # Dataset path (kept original)
    # DATASET_NAME = "人工智能1"  # Dataset name (example)
    # READING_FILE_LIST = [f"{DATASET_NAME}_train_val_test_5_fold_{num}.xlsx" for num in range(1, 6)]
    # WRITING_FILE = f"multi_label_dl_model_comparison_{DATASET_NAME}_5fold.xlsx"
    DATASET_NAME = "哪吒之魔童闹海的影评"  # Dataset name (English translation)

    READING_FILE_LIST = [f"{DATASET_NAME}_summary_version (deepseek-cot)_train_val_test_5_fold_{num}.xlsx" for num in range(1, 6)]
    WRITING_FILE = f"multi_label_dl_model_comparison_{DATASET_NAME}_summary_version (deepseek-cot)_5fold.xlsx"
    EPOCHS = 20  # Number of training epochs

    # Start training all models
    train_all_models_5folds(
        data_path=DATA_PATH,
        reading_file_list=READING_FILE_LIST,
        result_file=WRITING_FILE,
        epochs=EPOCHS
    )

