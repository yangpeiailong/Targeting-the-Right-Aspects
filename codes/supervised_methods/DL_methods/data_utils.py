import torch
import jieba
import openpyxl
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer  # Added: BERT tokenizer


# ------------------------------
# Dataset classes (distinguish between BERT and non-BERT)
# ------------------------------
class MyDataset(Dataset):
    """Base dataset (suitable for models like LSTM, CNN, FastText based on custom vocabularies)"""

    def __init__(self, x_data, y_data):
        self.x_data = x_data  # Text index sequence [batch_size, seq_len]
        self.y_data = y_data  # Labels [batch_size, label_num, label_class]

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return len(self.x_data)


class BertDataset(Dataset):
    """BERT-specific dataset (suitable for BERT series models, inputs include input_ids, etc.)"""

    def __init__(self, x_data, y_data):
        # x_data is a dictionary containing input_ids, attention_mask, token_type_ids
        self.input_ids = x_data['input_ids']
        self.attention_mask = x_data['attention_mask']
        self.token_type_ids = x_data['token_type_ids']
        self.y_data = y_data  # Labels [batch_size, label_num, label_class]

    def __getitem__(self, idx):
        # Return three inputs required by BERT + labels
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.y_data[idx]
        )

    def __len__(self):
        return len(self.y_data)


# ------------------------------
# Data loading and preprocessing (distinguish by model type)
# ------------------------------
def load_data(file_path, reading_file, model_type="base"):
    """
    Load data and perform preprocessing
    model_type: "base" (base models) or "bert" (BERT series models)
    """
    wb = openpyxl.load_workbook(f"{file_path}/{reading_file}")

    # Read texts and labels (general part)
    def read_sheet(sheet):
        texts, labels = [], []
        for row in sheet.iter_rows(values_only=True):
            texts.append(row[0])  # Raw text (not tokenized)
            labels.append(list(row[1:]))  # Label list
        return texts, labels

    x_train_raw, y_train = read_sheet(wb['train'])
    x_val_raw, y_val = read_sheet(wb['val'])
    x_test_raw, y_test = read_sheet(wb['test'])

    # Label processing (general: convert to one-hot encoding)
    def label2onehot(labels):
        all_labels = [e for row in labels for e in row]
        label2ind = {l: i for i, l in enumerate(set(all_labels))}
        label_num = len(labels[0]) if labels else 0  # Number of labels per sample
        label_class = len(label2ind)  # Number of classes per label
        # Convert to one-hot tensor [batch_size, label_num, label_class]
        onehot = torch.tensor([
            [[1 if label2ind[e] == c else 0 for c in label2ind.values()]
             for e in row] for row in labels
        ], dtype=torch.float)
        return onehot, label_num, label_class, label2ind

    y_train, label_num, label_class, label2ind = label2onehot(y_train)
    y_val, _, _, _ = label2onehot(y_val)  # Reuse training set's label2ind
    y_test, _, _, _ = label2onehot(y_test)

    # Process texts according to model type
    if model_type == "base":
        # Base models: custom tokenization + vocabulary (use training set's maxlen uniformly for multi-label adaptation)
        def process_text(texts, vocab=None, word2idx=None, maxlen=None, is_train=True):
            # Tokenization
            texts_cut = [jieba.lcut(text) for text in texts]

            if is_train:
                # Training set: calculate its own maxlen and build vocabulary
                maxlen = max(len(t) for t in texts_cut)
                # Build vocabulary (include all words in training set)
                vocab = list(set([w for t in texts_cut for w in t])) + ['<unk>', '<pad>']
                word2idx = {w: i for i, w in enumerate(vocab)}
            else:
                # Validation/test set: must use the training set's vocab, word2idx, and maxlen
                assert vocab is not None and word2idx is not None and maxlen is not None, \
                    "Validation/test set must pass in the training set's vocab, word2idx and maxlen"

            # Padding: uniformly use the training set's maxlen
            pad = '<pad>'
            texts_pad = [
                t + [pad] * (maxlen - len(t)) if len(t) < maxlen else t[:maxlen]
                for t in texts_cut
            ]

            # Convert texts to indices (use <unk> for OOV words)
            texts_idx = torch.tensor([
                [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in t]
                for t in texts_pad
            ])

            return texts_idx, vocab, word2idx, maxlen

        # 1. Process training set: calculate maxlen and build vocabulary
        x_train, vocab, word2idx, maxlen = process_text(x_train_raw, is_train=True)

        # 2. Process validation set: reuse training set parameters
        x_val, _, _, _ = process_text(x_val_raw, vocab=vocab, word2idx=word2idx, maxlen=maxlen, is_train=False)

        # 3. Process test set: reuse training set parameters
        x_test, _, _, _ = process_text(x_test_raw, vocab=vocab, word2idx=word2idx, maxlen=maxlen, is_train=False)

        return {
            'train': (x_train, y_train),
            'val': (x_val, y_val),
            'test': (x_test, y_test),
            'vocab': vocab,
            'word2idx': word2idx,
            'maxlen': maxlen,  # Unified as training set's maxlen
            'label_num': label_num,  # Keep label_num for multi-label scenario
            'label_class': label_class,
            'label2ind': label2ind
        }

    elif model_type == "bert":
        # BERT models: use BERT tokenizer (no need for custom vocabulary)
        def process_bert_text(texts, tokenizer, maxlen):
            # Process with BERT tokenizer
            encoding = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=maxlen,
                return_tensors='pt'  # Return torch tensors
            )
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'token_type_ids': encoding['token_type_ids']
            }

        # Load BERT tokenizer (must match the model, e.g., 'bert-base-chinese')
        tokenizer = BertTokenizer.from_pretrained('D:/bert-base-chinese')
        # Calculate suitable maxlen (take the 95th percentile of training set text lengths to avoid excessive length)
        text_lengths = [len(tokenizer.tokenize(text)) for text in x_train_raw]
        lengths_tensor = torch.tensor(text_lengths, dtype=torch.float32)
        maxlen = int(
            torch.tensor(lengths_tensor).quantile(0.95).item()) + 2  # +2 to reserve positions for [CLS] and [SEP]
        maxlen = min(maxlen, 512)  # BERT maximum length limit

        # Process training/validation/test sets
        x_train = process_bert_text(x_train_raw, tokenizer, maxlen)
        x_val = process_bert_text(x_val_raw, tokenizer, maxlen)
        x_test = process_bert_text(x_test_raw, tokenizer, maxlen)

        return {
            'train': (x_train, y_train),
            'val': (x_val, y_val),
            'test': (x_test, y_test),
            'tokenizer': tokenizer,  # Return tokenizer for subsequent use
            'maxlen': maxlen,
            'label_num': label_num,
            'label_class': label_class,
            'label2ind': label2ind
        }


# ------------------------------
# DataLoader generation (distinguish by model type)
# ------------------------------
def get_dataloaders(data, batch_size=32, model_type="base"):
    """Generate DataLoader, select dataset class according to model type"""
    if model_type == "base":
        train_dataset = MyDataset(data['train'][0], data['train'][1])
        val_dataset = MyDataset(data['val'][0], data['val'][1])
        test_dataset = MyDataset(data['test'][0], data['test'][1])
    elif model_type == "bert":
        # BERT's x is a dictionary containing input_ids, etc.
        train_dataset = BertDataset(data['train'][0], data['train'][1])
        val_dataset = BertDataset(data['val'][0], data['val'][1])
        test_dataset = BertDataset(data['test'][0], data['test'][1])

    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }