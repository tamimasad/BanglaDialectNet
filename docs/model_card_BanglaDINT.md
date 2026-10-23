---
language:
  - bn
  - en
library_name: transformers
pipeline_tag: translation
base_model: csebuetnlp/banglat5
license: cc-by-nc-sa-4.0
tags:
  - bangla
  - bengali
  - banglish
  - dialect-normalization
  - dialect-identification
  - machine-translation
  - multitask-learning
  - low-resource-nlp
  - t5
  - text2text-generation
---

# BanglaDINT-T5

A **Single T5** model that understands five Bangla regional dialects —
in both **native Bangla script and romanized Banglish** — and converts them to Standard
Bangla, English, or identifies the dialect itself. Fine-tuned from BanglaT5
on the published Vashantor benchmark, it **beats prior per-dialect specialist
baselines on every single dialect**, while being one generalist model
instead of five separate single-task ones.

**Highlights**

- Outperforms both published per-dialect baselines (mT5 and BanglaT5,
  Faria et al. 2023) on normalization BLEU, across all 5 dialects
- One model, 3 tasks: normalize -> Standard Bangla, translate -> English,
  identify the dialect
- Handles both Bangla script and romanized Banglish input natively
- Fully reproducible: data pipeline, training, and evaluation code are
  public (see [Reproducibility](#reproducibility))

## 🚀 Live Demo

Try this model directly in your browser, no setup required:
**[BanglaDINT-T5 Dialect Studio](https://huggingface.co/spaces/tamimaungan/BanglaDINT-T5-studio)**

## Quickstart

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from normalizer import normalize  # pip install git+https://github.com/csebuetnlp/normalizer

MODEL_ID = "tamimaungan/BanglaDINT-T5"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

text = normalize("আসো কোরোহম?")  # Barishal dialect

for task in ["normalize", "translate to english", "identify dialect"]:
    input_ids = tokenizer(f"{task}: {text}", return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=64, num_beams=4)
    print(task, "->", tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

Use the same 3 prefixes for Banglish input — just skip the `normalize()` call,
since that pipeline is for Bangla script only.

## Benchmarks

Tested on the Vashantor dataset's official held-out test split (1,869 rows,
5 dialects), never used during training or tuning.

### Normalization BLEU, vs. published baselines (Bangla script)

| Dialect       | mT5 (Faria et al.) | BanglaT5 (Faria et al.) | **BanglaDINT-T5** |
| ------------- | ------------------ | ----------------------- | ----------------- |
| Chittagong    | 36.75              | 44.03                   | **49.54**         |
| Noakhali      | 37.43              | 47.38                   | **54.05**         |
| Sylhet        | 51.32              | 51.08                   | **60.43**         |
| Barishal      | 48.56              | 53.50                   | **62.81**         |
| Mymensingh    | 64.74              | 69.06                   | **70.77**         |
| **Average\*** | 47.76              | 53.01                   | **59.52**         |

\* Average of the 5 published per-dialect scores, computed here — not an
overall figure reported by the original paper. mT5 and BanglaT5 above are
5 separate single-task models (one per dialect); BanglaDINT-T5 is one model
doing normalization, translation, and dialect ID jointly.

### Full results, both scripts

| Task                               | Bangla script | Banglish      |
| ---------------------------------- | ------------- | ------------- |
| Normalize (BLEU / chrF)            | 59.58 / 79.61 | 55.21 / 76.68 |
| Translate to English (BLEU / chrF) | 41.43 / 61.03 | 37.72 / 57.56 |
| Identify dialect (accuracy)        | 89.2%         | 87.0%         |

| Baseline                              | Normalize BLEU |
| ------------------------------------- | -------------- |
| Copy input unchanged                  | 10.52          |
| `banglat5` zero-shot (no fine-tuning) | 0.17           |

### Identify-dialect, per dialect

| Dialect    | Bangla | Banglish |
| ---------- | ------ | -------- |
| Barishal   | 97.6%  | 96.0%    |
| Mymensingh | 97.6%  | 94.9%    |
| Chittagong | 93.9%  | 90.7%    |
| Noakhali   | 85.6%  | 87.2%    |
| Sylhet     | 71.7%  | 66.4%    |

## Training

- Base: `csebuetnlp/banglat5` (T5, ~247M params), fine-tuned jointly on 6
  task variants (3 outputs x 2 input scripts)
- Data: train + validation combined (10,532 rows), with a 10% stratified
  slice held out for in-run checkpoint monitoring; the official test split
  was never used for tuning
- Per-example loss averaging (not per-token), so short targets (dialect
  labels) aren't under-weighted against long targets (full sentences)
- Checkpoint selection by a balanced score (dialect macro-accuracy +
  normalize chrF + translate chrF), not plain loss
- 8 epochs, bf16, fixed seed (42); evaluated with beam search (beam=4)

## Limitations

- Sylhet identification (~70%) is the weakest of the 5 dialects — likely
  reflects genuine ambiguity in how Sylheti reads once transcribed into
  standard Bangla script, not purely a model gap
- Recognizes only these 5 dialects; text from other dialects/languages
  isn't yet explicitly rejected (planned: confidence-based "Unknown"
  detection as a separate inference-time component, not a trained class)
- Translation BLEU (~38-41) is a solid first baseline, not state-of-the-art

## Reproducibility

Dataset pipeline, training script, and evaluation code:
[github.com/tamimasad/BanglaDialectNet](https://github.com/tamimasad/BanglaDialectNet)

## Dataset & Credits

Trained on the **Vashantor** dataset (Faria et al.), accessed via its
[Kaggle mirror](https://www.kaggle.com/datasets/munnavai/vashantor-dataset).
Fine-tuned from **BanglaT5** (Bhattacharjee et al., 2023), licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you
use this model, please cite both:

```bibtex
@misc{faria2023vashantor,
  title         = {Vashantor: A Large-scale Multilingual Benchmark Dataset for Automated Translation of Bangla Regional Dialects to Bangla Language},
  author        = {Fatema Tuj Johora Faria and Mukaffi Bin Moin and Ahmed Al Wase and Mehidi Ahmmed and Md. Rabius Sani and Tashreef Muhammad},
  year          = {2023},
  eprint        = {2311.11142},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@inproceedings{bhattacharjee-etal-2023-banglanlg,
  title     = "{B}angla{NLG} and {B}angla{T}5: Benchmarks and Resources for Evaluating Low-Resource Natural Language Generation in {B}angla",
  author    = "Bhattacharjee, Abhik and Hasan, Tahmid and Ahmad, Wasi Uddin and Shahriyar, Rifat",
  booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
  year      = "2023",
  pages     = "726--735",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2023.findings-eacl.54/"
}
```

## License

This model is a derivative of BanglaT5 and is released under the same
license as required by its ShareAlike terms:
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) —
attribution required, non-commercial use only, derivatives must carry the
same license.

## Author

Tamim Asad Aungan -- [GitHub](https://github.com/tamimasad) -- [LinkedIn](https://www.linkedin.com/in/tamim-asad-aungan-418ba53a6)
