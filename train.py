"""LoRA fine-tuning of a causal language model on Tamil stories.

Examples:
    python train.py                                   # distilgpt2, 1000 stories, 3 epochs
    python train.py --max-steps 20 --train-samples 64 # quick smoke run on CPU
    python train.py --push-to-hub your-name/distilgpt2-tamil-lora

Set HF_TOKEN in the environment (or a .env file) only if you push to the Hub.
"""

import json
import logging
import math
import os
from pathlib import Path

from tamil_lora.config import TrainConfig, parse_args
from tamil_lora.data import load_text_dataset, make_tokenize_fn

logger = logging.getLogger("train")


def train(config: TrainConfig) -> dict:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splits = load_text_dataset(
        config.dataset, config.text_column, config.train_samples, config.test_size, config.seed
    )
    tokenized = splits.map(
        make_tokenize_fn(tokenizer, config.text_column, config.max_length),
        batched=True,
        remove_columns=splits["train"].column_names,
    )
    logger.info("Train rows: %d, eval rows: %d", len(tokenized["train"]), len(tokenized["test"]))

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            fan_in_fan_out=model.config.model_type == "gpt2",  # GPT-2 uses Conv1D layers
        ),
    )
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        eval_strategy="epoch" if config.max_steps < 0 else "no",
        save_strategy="epoch" if config.max_steps < 0 else "no",
        save_total_limit=2,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    train_metrics = trainer.train().metrics
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"]) if eval_metrics["eval_loss"] < 50 else None

    output_dir = Path(config.output_dir)
    trainer.save_model(str(output_dir))  # adapter weights + adapter_config.json
    tokenizer.save_pretrained(str(output_dir))
    metrics = {"train": train_metrics, "eval": eval_metrics, "config": config.to_dict()}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logger.info("Saved adapter and metrics to %s", output_dir)

    if config.push_to_hub:
        token = os.environ.get("HF_TOKEN")
        model.push_to_hub(config.push_to_hub, token=token)
        tokenizer.push_to_hub(config.push_to_hub, token=token)
        logger.info("Pushed adapter to https://huggingface.co/%s", config.push_to_hub)
    return metrics


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    metrics = train(parse_args(argv))
    print(json.dumps(metrics["eval"], indent=2))


if __name__ == "__main__":
    main()
