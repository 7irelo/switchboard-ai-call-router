# Model card

**No encoder has been trained.** `train/train.py` and `intent/encoder.py` are written but have never been run, so they may need fixes on first use.

The service currently scores with `keyword-v1`, or with a TF-IDF model if one is supplied through `--model`. Neither has been evaluated on reportable data.

## Planned bake-off

Candidates: TF-IDF char n-grams (the baseline to beat), `Davlan/afro-xlmr-mini`, `Davlan/afro-xlmr-small`, `intfloat/multilingual-e5-small`, and `paraphrase-multilingual-MiniLM-L12-v2`. Licences and revision hashes must be pinned before use.

**Stop rule:** if no encoder beats TF-IDF on dev mean cost, ship TF-IDF and say so.

## To fill in after training

Dataset hash, git SHA, dev metrics per language mix, calibration (ECE before and after), size and p95 on pinned cores, and parity after quantisation.
