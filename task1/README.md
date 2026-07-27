# Task 1: Mountain Name NER

A named entity recognition model that finds mountain names in English text
(e.g. "Mount Everest" in *"We climbed Mount Everest last summer"*).

## Dataset

Built from **[Gepe55o/mountain-ner-dataset](https://huggingface.co/datasets/Gepe55o/mountain-ner-dataset)**
(HuggingFace Hub, ~111k labeled examples, sourced from NERetrieve and
Few-NERD). This dataset was chosen because it already provides labeled data
at this scale, removing the need for manual labeling or LLM-based generation
within the test's time constraints. The full process — splitting, converting
to BIO, and a label-quality check — is documented in `dataset_creation.ipynb`.

- **Splits:** train (79,757) / val (8,862) / test (22,110), saved as JSON
  Lines files in both binary and BIO format:
  `mountain_ner_{train,val,test}.jsonl` and
  `mountain_ner_{train,val,test}_bio.jsonl`.
- **Labeling formats:** binary (`O` / `MOUNTAIN`) is used for the final
  models; BIO (`O` / `B-MOUNTAIN` / `I-MOUNTAIN`) is also implemented and
  compared, since it can better separate adjacent mountain entities.
- **Known limitation:** ~6.61% of tokens tagged as "mountain" are lowercase,
  mostly from generic nouns (summit, peak, ridge, volcano) sometimes labeled
  even when used non-specifically. No manual filtering was applied, to keep
  the process reproducible and time-efficient — see the project report for
  this as a possible improvement.

## Models

Four models were fine-tuned (2 epochs each); results on the held-out test
split:

| Model | Scheme | Precision | Recall | F1 |
|---|---|---|---|---|
| **deberta-v3-base** (default) | binary | 0.833 | 0.853 | **0.843** |
| bert-base-cased | binary | 0.826 | 0.854 | 0.840 |
| bert-base-cased | bio | 0.824 | 0.854 | 0.839 |
| distilbert-base-cased | binary | 0.801 | 0.830 | 0.815 |

> Note: binary-scheme scores use seqeval's lenient evaluation (it doesn't
> recognize "MOUNTAIN" as a standard tag and merges adjacent tokens), while
> BIO is scored strictly. The two schemes aren't perfectly comparable on
> these numbers alone.

**Model weights** (HuggingFace Hub):
- [vl00835/mountain-ner-deberta-v3-base-binary](https://huggingface.co/vl00835/mountain-ner-deberta-v3-base-binary) (default)
- [vl00835/mountain-ner-bert-base-cased-binary](https://huggingface.co/vl00835/mountain-ner-bert-base-cased-binary)
- [vl00835/mountain-ner-bert-base-cased-bio](https://huggingface.co/vl00835/mountain-ner-bert-base-cased-bio)
- [vl00835/mountain-ner-distilbert-base-cased-binary](https://huggingface.co/vl00835/mountain-ner-distilbert-base-cased-binary)

## Setup

```bash
pip install -r requirements.txt
```

`dataset_creation.ipynb` and `demo.ipynb` were developed and run in Google
Colab. If run outside Colab, replace the Google Drive mounting cell with a
local path pointing to this folder.

## Usage

**Train** (reproduces one of the four models above):

```bash
python train.py --model_name microsoft/deberta-v3-base --label_scheme binary
python train.py --model_name bert-base-cased           --label_scheme binary
python train.py --model_name bert-base-cased           --label_scheme bio
python train.py --model_name distilbert-base-cased     --label_scheme binary
```

Each run saves its checkpoint to `models/<model>-<scheme>/final`.

**Run inference:**

```bash
python inference.py --text "We climbed Mount Everest last summer."
python inference.py --file article.txt --show_scores
python inference.py --model_path vl00835/mountain-ner-bert-base-cased-binary --text "..."
```

With no `--text`/`--file`, the script runs a few built-in demo sentences.
By default it loads `vl00835/mountain-ner-deberta-v3-base-binary` from the
HuggingFace Hub — no local checkpoint is required to try it.

See `demo.ipynb` for a walkthrough with more examples, including cases the
model struggles with.

## Potential improvements

See the project report (PDF) for proposed next steps, including manual
review of the generic-noun label noise and a full BIO-vs-binary comparison
for the DeBERTa model.
