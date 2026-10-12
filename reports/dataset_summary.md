# BanglaDialectNet -- Exploratory Data Analysis


## Train split (9375 rows)

- **standard_bangla**: chars mean=35.0 median=32 range=[9, 105] | words mean=6.7 range=[2, 19]
- **dialect_bangla**: chars mean=35.7 median=33 range=[8, 113] | words mean=6.7 range=[2, 22]
- **dialect_banglish**: chars mean=38.3 median=35 range=[8, 117] | words mean=6.8 range=[2, 22]
- **english**: chars mean=37.9 median=35 range=[7, 122] | words mean=7.7 range=[2, 26]

## Validation split (1250 rows)

- **standard_bangla**: chars mean=28.3 median=28 range=[12, 65] | words mean=5.5 range=[3, 13]
- **dialect_bangla**: chars mean=28.9 median=28 range=[10, 77] | words mean=5.5 range=[2, 14]
- **dialect_banglish**: chars mean=30.6 median=29 range=[11, 77] | words mean=5.6 range=[2, 14]
- **english**: chars mean=28.6 median=26 range=[8, 85] | words mean=6.1 range=[2, 16]

## Test split (1875 rows)

- **standard_bangla**: chars mean=38.3 median=38 range=[13, 92] | words mean=7.3 range=[3, 16]
- **dialect_bangla**: chars mean=38.3 median=38 range=[13, 100] | words mean=7.2 range=[3, 17]
- **dialect_banglish**: chars mean=41.1 median=40 range=[14, 100] | words mean=7.2 range=[3, 17]
- **english**: chars mean=39.8 median=39 range=[12, 110] | words mean=8.1 range=[3, 23]

## Unexpected characters (train split)

- '।' (DEVANAGARI DANDA): 56 occurrences
- '\u200c' (ZERO WIDTH NON-JOINER): 56 occurrences
- '\u200d' (ZERO WIDTH JOINER): 4 occurrences
- '/' (SOLIDUS): 2 occurrences
- ':' (COLON): 1 occurrences

## Unexpected characters by dialect input (train split)


### dialect_bangla

- '।' (DEVANAGARI DANDA): 42 occurrences
- '\u200c' (ZERO WIDTH NON-JOINER): 28 occurrences
- '\u200d' (ZERO WIDTH JOINER): 2 occurrences
- ':' (COLON): 1 occurrences
- '/' (SOLIDUS): 1 occurrences

### dialect_banglish

- '\u200c' (ZERO WIDTH NON-JOINER): 28 occurrences
- '\u200d' (ZERO WIDTH JOINER): 2 occurrences
- '/' (SOLIDUS): 1 occurrences

## Train/validation/test leakage check


### dialect_bangla -> standard_bangla

- None found -- no exact pair appears in more than one split.

### dialect_banglish -> standard_bangla

- None found -- no exact pair appears in more than one split.

## Identical dialect/standard text rows, by dialect (train split)


### dialect_bangla == standard_bangla

- Mymensingh: 80 rows
- Noakhali: 22 rows
- Barishal : 3 rows
- Sylhet: 3 rows

### dialect_banglish == standard Banglish

- Mymensingh: 59 rows
- Noakhali: 8 rows
- Barishal : 7 rows
- Sylhet: 2 rows