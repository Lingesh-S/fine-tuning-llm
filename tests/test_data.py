import pytest

from tamil_lora.data import make_tokenize_fn


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, texts, truncation, max_length):
        ids = [[len(word) for word in text.split()][:max_length] for text in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(i) for i in ids]}


def test_tokenize_appends_eos_and_truncates():
    tokenize = make_tokenize_fn(FakeTokenizer(), "text", max_length=3)
    out = tokenize({"text": ["ஒரு நாள்", "a b c d e"]})
    assert out["input_ids"][0] == [len("ஒரு"), len("நாள்<eos>")]
    assert len(out["input_ids"][1]) == 3


def test_local_text_dataset_split(tmp_path):
    pytest.importorskip("datasets")
    from tamil_lora.data import load_text_dataset

    path = tmp_path / "stories.txt"
    path.write_text("\n".join(f"கதை {i}" for i in range(40)) + "\n\n", encoding="utf-8")
    splits = load_text_dataset(str(path), "text", train_samples=20, test_size=0.25, seed=0)
    assert len(splits["train"]) == 15
    assert len(splits["test"]) == 5


def test_missing_column_is_reported(tmp_path):
    pytest.importorskip("datasets")
    from tamil_lora.data import load_text_dataset

    path = tmp_path / "rows.csv"
    path.write_text("story\nஒன்று\nஇரண்டு\n", encoding="utf-8")
    with pytest.raises(ValueError, match="story"):
        load_text_dataset(str(path), "text", train_samples=2, test_size=0.5, seed=0)
