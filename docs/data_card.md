# BanglaDialectNet -- Dataset Card

## Overview
Parallel dataset covering 5 Bangla dialects (Noakhali, Barishal, Chittagong, Sylhet, Mymensingh). Each row provides: standard Bangla, standard Bangla romanized (Banglish), dialect Bangla, dialect Bangla romanized, English translation, and a dialect label.

## Final split sizes
| Split | Rows |
|---|---|
| Train | 9295 |
| Validation | 1237 |
| Test | 1869 |

## Cleaning decisions
- 0 exact duplicate rows found (checked on every run)
- Dropped 99 rows total where dialect text exactly matched standard text AND dialect was Mymensingh (confirmed data-entry errors): Train=80, Validation=13, Test=6
- Kept identical-text rows for Noakhali/Barishal/Sylhet/Chittagong (confirmed genuinely identical short phrases -- valid training signal)
- No train/validation/test leakage detected (verified via exact-pair overlap check in EDA)

## Known characteristics
- Short sentences: ~6-8 words / 30-40 characters on average, max ~26 words
- Contains legitimate Bangla ZWJ/ZWNJ characters inside words (conjunct formation) -- preserved, not stripped
- See reports/dataset_summary.md and reports/figures/ for full EDA