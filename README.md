# 🚀 Fine-Tuning LLM for Tamil Story Generation

[![CI](https://github.com/Lingesh-S/fine-tuning-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/Lingesh-S/fine-tuning-llm/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lingesh-S/fine-tuning-llm/blob/main/colab_quickstart.ipynb)

Fine-tunes `distilgpt2` with **LoRA** (Low-Rank Adaptation) on Tamil short stories so it can continue a Tamil prompt such as **ஒரு நாள்** ("one day"). LoRA trains about 0.1% of the parameters, which keeps training fast and the saved adapter small. 📄 [LoRA paper](https://arxiv.org/abs/2106.09685)

---

# 📁 Project Structure

```bash
.
├── train.py                 # Load data, apply LoRA, train, evaluate, save adapter + metrics.json
├── generate.py              # Generate text with base model + adapter (or base model alone)
├── tamil_lora/
│   ├── config.py            # TrainConfig dataclass and CLI flags
│   └── data.py              # Dataset loading (Hub id or local file) and tokenisation
├── colab_quickstart.ipynb   # Runs train.py / generate.py on Colab
├── tests/                   # Unit tests + an offline end-to-end training smoke test
└── requirements.txt
```

# ⚡ What it does

* 🎯 Model: `distilgpt2` (any causal LM via `--model-name`)
* 🧠 Dataset: [`tniranjan/aitamilnadu_tamil_stories_no_instruct`](https://huggingface.co/datasets/tniranjan/aitamilnadu_tamil_stories_no_instruct) (first 1,000 stories by default), or a local `.txt` / `.jsonl` / `.csv`
* 💡 LoRA on GPT-2 attention (`c_attn`, r=8, alpha=32, dropout=0.1) with the Hugging Face `Trainer`
* 📏 Evaluation loss and perplexity each epoch, saved to `outputs/.../metrics.json`
* ☁️ Optional `--push-to-hub` (uses `HF_TOKEN`)

# 📌 How to Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py                                       # full run (GPU recommended)
python train.py --max-steps 20 --train-samples 64     # quick CPU smoke run
python generate.py --adapter outputs/distilgpt2-tamil-lora --prompt "ஒரு நாள் "
python generate.py --no-adapter --prompt "ஒரு நாள் "  # compare with the base model
```

`python train.py --help` lists every option. Or open the notebook in Colab with the badge above.

# 🧪 Tests

```bash
pip install -r requirements-dev.txt
ruff check . && pytest
```

The smoke test builds a tiny random GPT-2 and tokenizer on the fly and runs `train.py` for 3 steps, so it needs no downloads.

# 🧠 Concepts Demonstrated

* Parameter-efficient fine-tuning of a causal language model with LoRA
* Tokenisation without wasted padding (dynamic padding in the data collator)
* Reproducible runs: fixed seed, saved config and metrics
* Model saving/loading and sampling-based generation

# ⚠️ Limitations

distilgpt2's tokenizer was built for English, so Tamil text splits into many byte-level tokens and output quality is limited. A multilingual base model (for example a small Llama, Qwen or Gemma variant) is the natural next step: pass it with `--model-name` and set `--target-modules q_proj v_proj`.

## 📚 Learning Resources

If you're exploring LoRA (Low-Rank Adaptation) or PEFT (Parameter-Efficient Fine-Tuning), here are some high-quality resources to deepen your understanding:

- 🔬 [LoRA: Low-Rank Adaptation of Large Language Models (Paper)](https://arxiv.org/abs/2106.09685)
- 🤗 [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index)
- 💡 [Fine-Tuning vs. Parameter-Efficient Fine-Tuning – Sebastian Raschka](https://sebastianraschka.com/blog/2023/peft.html)

These links complement the methods used in `train.py`.


# 🙌 Acknowledgements

This repo demonstrates a practical application of Hugging Face Transformers and LoRA. Special thanks to open datasets and pretrained models shared on 🤗 Hugging Face Hub.

# 👨‍💻 Author

**Lingesh S**
Interested in LLMs, RAG pipelines, and practical NLP

[GitHub](https://github.com/Lingesh-S) • [LinkedIn](https://linkedin.com/in/lingesh-s29)
