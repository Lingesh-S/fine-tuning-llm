"""Generate Tamil text with a base model plus a trained LoRA adapter.

    python generate.py --adapter outputs/distilgpt2-tamil-lora --prompt "ஒரு நாள்"
    python generate.py --no-adapter --prompt "ஒரு நாள்"     # base model, for comparison
"""

import argparse


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", default="outputs/distilgpt2-tamil-lora", help="Local folder or Hub repo id")
    parser.add_argument("--base-model", default=None, help="Defaults to the adapter's base model")
    parser.add_argument("--no-adapter", action="store_true", help="Use the base model only")
    parser.add_argument("--prompt", default="ஒரு நாள் ")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    base_name = args.base_model or (
        "distilgpt2" if args.no_adapter else PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForCausalLM.from_pretrained(base_name)
    if not args.no_adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
