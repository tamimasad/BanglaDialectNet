# BanglaDialectNet

**BanglaDINT-T5** — a single multi-task T5 model that converts Bangla regional
dialect speech/text (Bangla script _or_ romanized Banglish) into **Standard
Bangla**, **English**, and identifies **which of 5 dialects** it was written in:
Noakhali, Barishal, Chittagong, Sylhet, and Mymensingh.

[![Try the demo](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/tamimaungan/BanglaDINT-T5-studio)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-BanglaDINT--T5-yellow)](https://huggingface.co/tamimaungan/BanglaDINT-T5)
[![Code License: Apache 2.0](https://img.shields.io/badge/code%20license-Apache--2.0-blue)](LICENSE)

## Live demo

Try it directly in your browser, no setup required:
**[huggingface.co/spaces/tamimaungan/BanglaDINT-T5-studio](https://huggingface.co/spaces/tamimaungan/BanglaDINT-T5-studio)**

Type a sentence in any of the 5 dialects (Bangla script or Banglish) and get
Standard Bangla, an English translation, and the detected dialect, with a PDF
export of your session.

## Model

The trained model is published on the Hugging Face Hub:
**[huggingface.co/tamimaungan/BanglaDINT-T5](https://huggingface.co/tamimaungan/BanglaDINT-T5)**

Full benchmark tables, training configuration, limitations, and citations are
in the [model card](docs/model_card_BanglaDINT.md) (same content as the Hub
page).

Fine-tuned from [csebuetnlp/banglat5](https://huggingface.co/csebuetnlp/banglat5)
on the [Vashantor](https://arxiv.org/abs/2311.11142) dialect benchmark,
BanglaDINT-T5 beats the paper's published per-dialect specialist baselines on
normalization BLEU for every one of the 5 dialects, while being a single
generalist model doing 3 tasks at once rather than 5 separate single-task
models.

## How it works

One model, six task variants, selected by a task-prefixed input (the same
trick T5 itself uses):

| Task                         | Bangla-script prefix    | Banglish prefix                  |
| ---------------------------- | ----------------------- | -------------------------------- |
| Normalize to Standard Bangla | `normalize:`            | `normalize banglish:`            |
| Translate to English         | `translate to english:` | `translate to english banglish:` |
| Identify dialect             | `identify dialect:`     | `identify dialect banglish:`     |

## Project structure

```
├── src/data_loader/   # dataset loading, cleaning, multi-task example building
├── scripts/           # training, evaluation, diagnostics, publishing
├── docs/               # data card, model card, results write-ups
├── reports/            # EDA and dataset summary reports
├── space/              # the Gradio app behind the live demo (Hub Space)
└── models/             # trained model artifacts (not committed, hosted on the Hub)
```

## Reproducing the pipeline

```bash
git clone https://github.com/tamimasad/BanglaDialectNet.git
cd BanglaDialectNet
pip install -r requirements.txt
```

### 1. Get the dataset

This project trains on the **Vashantor** benchmark, available on its
[Kaggle mirror](https://www.kaggle.com/datasets/munnavai/vashantor-dataset).
Download it, then arrange the files under `dataset/raw/` so it looks like
this (one CSV per dialect inside each split folder — rename/reorganize the
downloaded folders to match if Kaggle's own structure differs):

dataset/raw/
├── Train/ # 5 per-dialect CSVs
├── Validation/ # 5 per-dialect CSVs
└── Test/ # 5 per-dialect CSVs

`dataset/` is git-ignored on purpose — the raw data isn't redistributed in
this repo, only pulled from Kaggle directly.

### 2. Clean and process it

```bash
python scripts/clean_dataset.py
```

Writes cleaned, per-dialect CSVs to `dataset/processed/{Train,Validation,Test}/`,
applying the cleaning decisions documented in [docs/data_card.md](docs/data_card.md)
(for example, dropping 99 Mymensingh rows confirmed as data-entry errors).

### 3. (Optional) regenerate the EDA report

```bash
python scripts/dataset_report.py
```

### 4. Train the final, balanced model

```bash
python scripts/train_bangladint.py
```

### 5. Evaluate once, on the held-out test set

```bash
python scripts/evaluate_banglat5.py --model models/BanglaDINT-T5-production --split test
```

## Dataset & credits

Built on the **Vashantor** benchmark:

> Faria, F. T. J., et al. _"Vashantor: A Large-scale Multilingual Benchmark
> Dataset for Automated Translation of Bangla Regional Dialects to Bangla
> Language."_ 2023. [arXiv:2311.11142](https://arxiv.org/abs/2311.11142)

Base model: [csebuetnlp/banglat5](https://huggingface.co/csebuetnlp/banglat5),
from the [BanglaNLG paper](https://arxiv.org/abs/2205.11081), CC BY-NC-SA 4.0.

## License

Code in this repository: [Apache 2.0](LICENSE).
Model weights: CC BY-NC-SA 4.0 (inherited from the base model, see the
[model card](docs/model_card_BanglaDINT.md) for full terms).

## Author

**Tamim Asad Aungan**
[GitHub](https://github.com/tamimasad) ·
[Hugging Face](https://huggingface.co/tamimaungan) ·
[LinkedIn](https://www.linkedin.com/in/tamim-asad-aungan-418ba53a6)
