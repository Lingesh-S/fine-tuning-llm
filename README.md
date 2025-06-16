# 🚀 Fine-Tuning LLM for Tamil Story Generation

This project demonstrates fine-tuning of transformer-based language models for creative Tamil story generation. It includes two parts:

* 📘 `Fine_Tune_LLM_Part_1.ipynb` – Basic fine-tuning using Hugging Face Transformers
* 🧪 `Fine_Tune_LLM_Part_2.py` – Advanced fine-tuning with LoRA (Low-Rank Adaptation) for efficient training

---

# 📁 Project Structure

```bash
.
├── Fine_Tune_LLM_Part_1.ipynb   # Basic fine-tuning using HuggingFace Trainer
├── Fine_Tune_LLM_Part_2.py      # Full pipeline with dataset prep, LoRA, and model saving
└── README.md                    # You're here!
```

# 📌 Part 1: Basic LLM Fine-Tuning

* 🔧 Model: llama-2-7b
* 📦 Library: transformers
* 📊 Dataset: Custom via `load_dataset("your_dataset")`

## ✨ Key Steps

* Load tokenizer and model
* Tokenize dataset
* Define training configuration
* Use Hugging Face Trainer API
* Save the fine-tuned model

# ⚡ Part 2: LoRA-based Efficient Fine-Tuning

* 🎯 Goal: Fine-tune distilgpt2 on Tamil stories
* 🧠 Dataset: `tniranjan/aitamilnadu_tamil_stories_no_instruct`
* 💡 Optimization: LoRA + AdamW + Trainer API
* 🔐 Uses Hugging Face Hub login via `.env`

## 🔧 Key Libraries Used

* `transformers`
* `peft` (Parameter-Efficient Fine-Tuning)
* `datasets`, `torch`, `pandas`
* `huggingface_hub`, `python-dotenv`

## 📘 Highlights

* Preprocessing and tokenization
* LoRA configuration via `LoraConfig`
* Custom training loop using `Trainer`
* Saves model to Google Drive (Colab integration)
* Generates creative Tamil stories using prompts like **ஒரு நாள்**

# 📌 How to Run

▶️ Open Notebook in Colab

# 🧠 Concepts Demonstrated

* Transfer Learning in NLP
* Fine-tuning Causal Language Models
* Tokenization and padding
* Training configuration with `TrainingArguments`
* Efficient training using LoRA (low-rank matrix adaptation)
* Model saving/loading and inference
* Tamil language generation with Transformers

# 📦 Requirements

```bash
pip install datasets pandas torch transformers[torch] python-dotenv peft
```

# 📂 Outputs

* 🔤 Fine-tuned model saved to local or Google Drive
* 📜 Text generations in Tamil using seed prompts

# 🙌 Acknowledgements

This repo demonstrates a practical application of Hugging Face Transformers and LoRA. Special thanks to open datasets and pretrained models shared on 🤗 Hugging Face Hub.

# 👨‍💻 Author

**Lingesh S**
Interested in LLMs, RAG pipelines, and practical NLP
[GitHub](https://github.com/Lingesh-S) • [LinkedIn](https://linkedin.com/in/lingesh-s29)
