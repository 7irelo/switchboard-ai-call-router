# Model card

**No encoder has been evaluated.** The training and export code has run once as a code check: `paraphrase-multilingual-MiniLM-L12-v2`, two CPU epochs over 600 template rows. It proved training, ONNX export, INT8 quantisation, vocabulary trimming, parity checks and serving all work, and its figures are in `docs/REPORT.md`. Its accuracy means nothing, and it isn't served.

The service scores with `keyword-v1`, or the TF-IDF model from `poe train-tfidf`. TF-IDF is calibrated on real Whisper transcripts when they exist, because fitting on fixture text pushed temperature to its lower bound.

## Bake-off

The procedure has run on CPU over 600 template rows (`train/bakeoff.py`, results in `docs/REPORT.md`). The encoders were badly underfit, so TF-IDF's lead there says nothing about encoders. The candidates still to run fairly:

Candidates: TF-IDF char n-grams (the baseline to beat), `Davlan/afro-xlmr-mini`, `Davlan/afro-xlmr-small`, `intfloat/multilingual-e5-small`, and `paraphrase-multilingual-MiniLM-L12-v2`. Licences and revision hashes must be pinned before use.

**Stop rule:** if no encoder beats TF-IDF on dev mean cost, ship TF-IDF and say so.

## To fill in after training

Dataset hash, git SHA, dev metrics per language mix, calibration (ECE before and after), size and p95 on pinned cores, and parity after quantisation.
