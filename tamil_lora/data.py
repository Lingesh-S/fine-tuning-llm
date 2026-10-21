"""Dataset loading and tokenisation."""

from pathlib import Path


def load_text_dataset(name_or_path: str, text_column: str, train_samples: int, test_size: float, seed: int):
    """Load up to `train_samples` rows and split them into train/test.

    Accepts a Hub dataset id or a local .txt (one example per line),
    .jsonl/.json or .csv file.
    """
    from datasets import load_dataset

    path = Path(name_or_path)
    if path.exists():
        builder = {".txt": "text", ".jsonl": "json", ".json": "json", ".csv": "csv"}.get(path.suffix.lower())
        if builder is None:
            raise ValueError(f"Unsupported local dataset type: {path.suffix}")
        dataset = load_dataset(builder, data_files=str(path), split="train")
    else:
        dataset = load_dataset(name_or_path, split="train")

    if text_column not in dataset.column_names:
        raise ValueError(f"Column '{text_column}' not found; available: {dataset.column_names}")
    dataset = dataset.filter(lambda row: bool(row[text_column] and row[text_column].strip()))
    dataset = dataset.select(range(min(train_samples, len(dataset))))
    return dataset.train_test_split(test_size=test_size, seed=seed)


def make_tokenize_fn(tokenizer, text_column: str, max_length: int):
    """Batched tokeniser for `Dataset.map`. Padding is left to the data collator."""

    def tokenize(batch):
        texts = [text + tokenizer.eos_token for text in batch[text_column]]
        return tokenizer(texts, truncation=True, max_length=max_length)

    return tokenize
