"""
Fine-tune a BERT-based model for mountain-name NER.

Supports both labeling schemes produced by dataset_creation.ipynb:
  * binary : tags 0 / 1                      -> 2 classes (O, MOUNTAIN)
  * bio    : tags O / B-MOUNTAIN / I-MOUNTAIN -> 3 classes

All four reported experiments were
produced with:

    python train.py --model_name microsoft/deberta-v3-base --label_scheme binary
    python train.py --model_name bert-base-cased           --label_scheme binary
    python train.py --model_name bert-base-cased           --label_scheme bio
    python train.py --model_name distilbert-base-cased     --label_scheme binary

"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# Label schemes
LABEL_SCHEMES = {
    # binary: the original dataset labels (0 = not a mountain, 1 = mountain)
    "binary": {
        "labels": ["O", "MOUNTAIN"],
        "file_suffix": "",
        "tag_column": "tags",
    },
    # bio: standard NER scheme, distinguishes entity start from continuation
    "bio": {
        "labels": ["O", "B-MOUNTAIN", "I-MOUNTAIN"],
        "file_suffix": "_bio",
        "tag_column": "bio_tags",
    },
}


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an encoder for mountain NER")

    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base",
                        help="Any token-classification-capable model from the HF Hub. "
                             "The default reproduces the final model; the other "
                             "reported experiments used bert-base-cased and "
                             "distilbert-base-cased.")
    parser.add_argument("--label_scheme", type=str, default="binary",
                        choices=list(LABEL_SCHEMES),
                        help="Which labeling scheme (and which .jsonl files) to use")

    parser.add_argument("--data_dir", type=str, default="data",
                        help="Folder containing the .jsonl dataset files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save checkpoints. Defaults to "
                             "models/<model>-<scheme>")

    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length in subword tokens")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size (lower it if you run out of memory)")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    # memory-saving options
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps. Effective batch size = "
                             "batch_size * grad_accum, without extra memory cost.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Trade ~30%% speed for a large drop in activation memory")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        choices=["adamw_torch", "adafactor"],
                        help="adafactor uses far less memory than AdamW")

    parser.add_argument("--train_subset", type=int, default=0,
                        help="Train on a random subset of N examples (0 = use all data)")
    parser.add_argument("--val_subset", type=int, default=0,
                        help="Evaluate on a random subset of N examples (0 = use all)")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip the final evaluation on the test split")

    return parser.parse_args()


# Data loading
def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_splits(data_dir, scheme):
    """Load train/val/test splits and normalise labels to integer ids."""
    config = LABEL_SCHEMES[scheme]
    label2id = {label: i for i, label in enumerate(config["labels"])}
    suffix = config["file_suffix"]
    column = config["tag_column"]

    files = {
        "train": os.path.join(data_dir, f"mountain_ner_train{suffix}.jsonl"),
        "validation": os.path.join(data_dir, f"mountain_ner_val{suffix}.jsonl"),
        "test": os.path.join(data_dir, f"mountain_ner_test{suffix}.jsonl"),
    }

    splits = {}
    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {name} file: {path}\n"
                f"Run dataset_creation_Task-1.ipynb first, or pass a different --data_dir."
            )

        records = []
        for record in load_jsonl(path):
            tags = record[column]
            if tags and isinstance(tags[0], str):     # BIO files store strings
                tags = [label2id[tag] for tag in tags]
            records.append({"tokens": record["tokens"], "tags": tags})

        splits[name] = Dataset.from_list(records)

    return splits


# Tokenization with label alignment
def build_tokenize_fn(tokenizer, max_length):
    """Return a function that tokenizes words and aligns word-level labels to subwords.

    Transformer tokenizers split rare words into several subword tokens, so the
    number of tokens no longer matches the number of word-level labels. The real
    label is assigned to the first subword of each word, and -100 to everything
    else (special tokens, subword continuations, padding). PyTorch ignores -100
    when computing the loss, so each word contributes exactly once.
    """

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,   # input is already a list of words
            truncation=True,
            max_length=max_length,
        )

        all_labels = []
        for i, labels in enumerate(examples["tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            previous_word_id = None
            label_ids = []

            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)              # [CLS], [SEP], padding
                elif word_id != previous_word_id:
                    label_ids.append(labels[word_id])   # first subword of a word
                else:
                    label_ids.append(-100)              # continuation subword
                previous_word_id = word_id

            all_labels.append(label_ids)

        tokenized["labels"] = all_labels
        return tokenized

    return tokenize_and_align_labels


# Metrics
def build_compute_metrics(label_list):
    """Return a metric function for the given label list."""

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        true_labels, true_predictions = [], []
        for prediction, label in zip(predictions, labels):
            current_labels, current_preds = [], []
            for p, l in zip(prediction, label):
                if l != -100:
                    current_labels.append(label_list[l])
                    current_preds.append(label_list[p])
            true_labels.append(current_labels)
            true_predictions.append(current_preds)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    return compute_metrics


# Device detection
def describe_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"Device: CUDA GPU - {name} ({memory_gb:.1f} GB)")
        return True                      # fp16 is supported on CUDA
    if torch.backends.mps.is_available():
        print("Device: Apple MPS (Metal)")
        return False                     # fp16 via Trainer is not supported on MPS
    print(f"Device: CPU only ({os.cpu_count()} cores) - training will be slow")
    return False


def main():
    args = parse_args()
    use_fp16 = describe_device()

    label_list = LABEL_SCHEMES[args.label_scheme]["labels"]
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    output_dir = args.output_dir or os.path.join(
        "models", f"{args.model_name.split('/')[-1]}-{args.label_scheme}"
    )

    print(f"\nModel:        {args.model_name}")
    print(f"Label scheme: {args.label_scheme} ({len(label_list)} classes: "
          f"{', '.join(label_list)})")
    print(f"Output dir:   {output_dir}")

    # Data
    print(f"\nLoading data from: {args.data_dir}")
    splits = load_splits(args.data_dir, args.label_scheme)

    if args.train_subset:
        splits["train"] = splits["train"].shuffle(seed=args.seed).select(
            range(min(args.train_subset, len(splits["train"])))
        )
    if args.val_subset:
        splits["validation"] = splits["validation"].shuffle(seed=args.seed).select(
            range(min(args.val_subset, len(splits["validation"])))
        )

    print(f"Train: {len(splits['train'])}, "
          f"Val: {len(splits['validation'])}, "
          f"Test: {len(splits['test'])}")

    # Tokenization
    # Every architecture has its own tokenizer, so the data is tokenized and the
    # labels realigned for whichever model was requested.
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenize_fn = build_tokenize_fn(tokenizer, args.max_length)

    tokenized = {
        name: ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
        for name, ds in splits.items()
    }

    # Model
    # Some checkpoints (notably DeBERTa-v3) are stored in fp16, which breaks
    # gradient scaling when fp16 training is enabled. Loading in fp32 explicitly
    # is a no-op for fp32 checkpoints and fixes the fp16 ones.
    print(f"Loading model: {args.model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        dtype=torch.float32,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Training setup
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=200,
        fp16=use_fp16,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(label_list),
    )

    # Train
    print("\nStarting training...\n")
    trainer.train()

    # Save
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)          # weights + config (incl. id2label)
    tokenizer.save_pretrained(final_dir)   # tokenizer must travel with the model
    print(f"\nModel saved to: {final_dir}")

    # Final evaluation
    if not args.skip_test:
        print("\nEvaluating on the held-out test split...")
        results = trainer.evaluate(tokenized["test"])

        print("\nTest results:")
        for key in ("eval_precision", "eval_recall", "eval_f1", "eval_loss"):
            if key in results:
                print(f"  {key.replace('eval_', ''):<10} {results[key]:.4f}")

        results["model_name"] = args.model_name
        results["label_scheme"] = args.label_scheme
        with open(os.path.join(final_dir, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
