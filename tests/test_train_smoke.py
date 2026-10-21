"""End-to-end run of train.py on a tiny random GPT-2, fully offline, on CPU."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

STORIES = [
    "ஒரு நாள் ஒரு சிறுவன் காட்டுக்குச் சென்றான்",
    "அவன் ஒரு யானையைப் பார்த்தான்",
    "யானை அவனுடன் நண்பனாக மாறியது",
    "அவர்கள் ஆற்றங்கரையில் விளையாடினர்",
] * 8


@pytest.fixture
def tiny_model(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(STORIES, trainers.WordLevelTrainer(special_tokens=["[UNK]", "<|endoftext|>"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="<|endoftext|>")

    model_dir = tmp_path / "tiny-gpt2"
    fast.save_pretrained(model_dir)
    config = GPT2Config(vocab_size=len(fast), n_positions=64, n_embd=32, n_layer=2, n_head=2)
    GPT2LMHeadModel(config).save_pretrained(model_dir)

    data = tmp_path / "stories.txt"
    data.write_text("\n".join(STORIES), encoding="utf-8")
    return model_dir, data


def test_train_runs_and_saves_adapter(tiny_model, tmp_path):
    from train import main

    model_dir, data = tiny_model
    out = tmp_path / "out"
    main([
        "--model-name", str(model_dir), "--dataset", str(data), "--train-samples", "32",
        "--test-size", "0.25", "--max-length", "16", "--batch-size", "4", "--max-steps", "3",
        "--output-dir", str(out),
    ])
    assert (out / "adapter_config.json").exists()
    assert any(out.glob("adapter_model.*"))
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["train"]["train_loss"] > 0
    assert "eval_loss" in metrics["eval"]
