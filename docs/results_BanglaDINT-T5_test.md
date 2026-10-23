# BanglaDialectNet -- BanglaDINT-T5 Results (test set)
## Bangla-script vs Banglish input, side by side

## Task: normalize
- Bangla script: BLEU=59.58  chrF=79.61  exact-match=26.9%
    - Barishal: BLEU=62.81  chrF=81.47
    - Chittagong: BLEU=49.54  chrF=73.68
    - Mymensingh: BLEU=70.77  chrF=87.00
    - Noakhali: BLEU=54.05  chrF=76.32
    - Sylhet: BLEU=60.43  chrF=79.63
    (copy baseline, no model: BLEU=10.52  chrF=43.56)
- Banglish: BLEU=55.21  chrF=76.68  exact-match=23.3%
    - Barishal: BLEU=59.57  chrF=79.30
    - Chittagong: BLEU=47.70  chrF=72.14
    - Mymensingh: BLEU=65.28  chrF=83.30
    - Noakhali: BLEU=49.84  chrF=73.40
    - Sylhet: BLEU=53.38  chrF=75.30

## Task: translate_to_english
- Bangla script: BLEU=41.43  chrF=61.03  exact-match=13.1%
    - Barishal: BLEU=44.32  chrF=63.19
    - Chittagong: BLEU=35.71  chrF=56.06
    - Mymensingh: BLEU=45.62  chrF=65.08
    - Noakhali: BLEU=39.44  chrF=59.29
    - Sylhet: BLEU=41.98  chrF=61.58
- Banglish: BLEU=37.72  chrF=57.56  exact-match=12.3%
    - Barishal: BLEU=40.69  chrF=59.92
    - Chittagong: BLEU=32.87  chrF=53.31
    - Mymensingh: BLEU=41.48  chrF=61.53
    - Noakhali: BLEU=36.02  chrF=55.65
    - Sylhet: BLEU=37.48  chrF=57.46

## Task: identify_dialect
- Bangla script: accuracy=89.2%
    - Barishal: 97.6% (366/375)
    - Chittagong: 93.9% (352/375)
    - Mymensingh: 97.6% (360/369)
    - Noakhali: 85.6% (321/375)
    - Sylhet: 71.7% (269/375)
- Banglish: accuracy=87.0%
    - Barishal: 96.0% (360/375)
    - Chittagong: 90.7% (340/375)
    - Mymensingh: 94.9% (350/369)
    - Noakhali: 87.2% (327/375)
    - Sylhet: 66.4% (249/375)
