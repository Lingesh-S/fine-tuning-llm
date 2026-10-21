"""Training configuration and its command-line interface."""

import argparse
from dataclasses import asdict, dataclass, field


@dataclass
class TrainConfig:
    model_name: str = "distilgpt2"
    dataset: str = "tniranjan/aitamilnadu_tamil_stories_no_instruct"
    text_column: str = "text"
    train_samples: int = 1000
    test_size: float = 0.05
    max_length: int = 256
    epochs: float = 3.0
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["c_attn"])
    output_dir: str = "outputs/distilgpt2-tamil-lora"
    max_steps: int = -1
    seed: int = 42
    push_to_hub: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="LoRA fine-tune a causal LM on Tamil stories.")
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument(
        "--dataset",
        default=defaults.dataset,
        help="Hugging Face dataset id, or a local .txt/.jsonl/.csv file",
    )
    parser.add_argument("--text-column", default=defaults.text_column)
    parser.add_argument("--train-samples", type=int, default=defaults.train_samples)
    parser.add_argument("--test-size", type=float, default=defaults.test_size)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--epochs", type=float, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--lora-r", type=int, default=defaults.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--target-modules", nargs="+", default=defaults.target_modules)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps, help="Stop early (smoke tests)")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID", help="Push the adapter to this Hub repo")
    args = parser.parse_args(argv)

    config = TrainConfig(**{k: v for k, v in vars(args).items()})
    if not 0 < config.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    if config.train_samples < 2:
        parser.error("--train-samples must be at least 2")
    return config
