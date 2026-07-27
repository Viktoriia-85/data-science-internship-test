"""
Run mountain-name NER inference with a fine-tuned model.

Works with both labeling schemes: the label names are read from the model's own
config, so binary (O / MOUNTAIN) and BIO (O / B-MOUNTAIN / I-MOUNTAIN) checkpoints
are both handled without extra flags.

Available checkpoints (see train.py for the commands that produced them):
    models/deberta-v3-base-binary/final    - best result, F1 0.843 (default)
    models/bert-base-cased-binary/final
    models/bert-base-cased-bio/final
    models/distilbert-base-cased-binary/final

Examples:
    python inference.py --text "We climbed Mount Everest last summer."
    python inference.py --file article.txt --show_scores
    python inference.py --model_path models/bert-base-cased-binary/final --text "..."

With no --text and no --file, the script runs a few built-in demo sentences.
"""

import argparse
import json
import os
import string

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

DEFAULT_MODEL_PATH = "models/deberta-v3-base-binary/final"

DEMO_SENTENCES = [
    "We climbed Mount Everest last summer and later visited Kilimanjaro in Tanzania.",
    "The Alps stretch across eight countries in Europe.",
    "She works as a data scientist in Kyiv and enjoys hiking.",
    "K2 is considered more dangerous to climb than Everest.",
    "The summit of Ben Nevis is often covered in clouds.",
]


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Find mountain names in text")

    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Directory with the fine-tuned model and its tokenizer, "
                             "or a model id on the HuggingFace Hub")

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", type=str, help="Text to analyse")
    source.add_argument("--file", type=str, help="Path to a UTF-8 text file to analyse")

    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length in subword tokens")
    parser.add_argument("--show_scores", action="store_true",
                        help="Print the model's confidence for each predicted word")
    parser.add_argument("--json", action="store_true",
                        help="Print results as JSON instead of human-readable text")

    return parser.parse_args()


# Model
class MountainNER:

    def __init__(self, model_path, max_length=256):
        if not os.path.isdir(model_path) and "/" not in model_path:
            raise FileNotFoundError(
                f"Model directory not found: {model_path}\n"
                f"Train a model first (python train.py), or pass --model_path."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.max_length = max_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()   # disable dropout — inference must be deterministic

        self.id2label = self.model.config.id2label

    def predict_words(self, text):
        """Return (word, label, confidence) for every word in the text.

        The training data was tokenized with a simple whitespace split, so the same
        splitting is used here to keep inference consistent with training.
        """
        words = text.split()
        if not words:
            return []

        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoded).logits

        probabilities = torch.softmax(logits, dim=-1)[0]
        predictions = logits.argmax(dim=-1)[0].tolist()
        word_ids = encoded.word_ids()

        # Read the prediction from the first subtoken of each word — the same
        # convention used when aligning labels during training.
        results = []
        previous_word_id = None
        for position, (prediction, word_id) in enumerate(zip(predictions, word_ids)):
            if word_id is not None and word_id != previous_word_id:
                score = probabilities[position][prediction].item()
                results.append((words[word_id], self.id2label[prediction], score))
            previous_word_id = word_id

        return results

    def extract_mountains(self, text):
        """Group consecutive mountain-labeled words into full entity names.

        Punctuation is stripped from words like "Everest." - a side effect of
        whitespace-split training data - so identical entities aren't counted
        as different.
        """
        entities = []
        current = []
        current_scores = []

        def flush():
            if current:
                name = " ".join(current).strip(string.punctuation)
                if name:
                    entities.append({
                        "name": name,
                        "confidence": round(min(current_scores), 4),
                    })

        for word, label, score in self.predict_words(text):
            if label == "O":
                flush()
                current.clear()
                current_scores.clear()
            else:
                # In the BIO scheme a B- tag starts a new entity even when the
                # previous word was also part of one.
                if label.startswith("B-") and current:
                    flush()
                    current.clear()
                    current_scores.clear()
                current.append(word)
                current_scores.append(score)

        flush()                        # flush the last entity if the text ends on one
        return entities


# Output helpers
def print_result(text, entities, ner=None, show_scores=False):
    print(f"\n{text}")

    if entities:
        for entity in entities:
            print(f"  -> {entity['name']}  (confidence {entity['confidence']:.3f})")
    else:
        print("  -> no mountains found")

    if show_scores and ner is not None:
        print("  word-level predictions:")
        for word, label, score in ner.predict_words(text):
            if label != "O":
                print(f"     {word:<20} {label:<12} {score:.3f}")


def main():
    args = parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.text:
        texts = [args.text]
    else:
        print("No --text or --file given, running demo sentences.")
        texts = DEMO_SENTENCES

    ner = MountainNER(args.model_path, max_length=args.max_length)

    if not args.json:
        print(f"Model:  {args.model_path}")
        print(f"Device: {ner.device}")
        print(f"Labels: {ner.id2label}")

    output = []
    for text in texts:
        entities = ner.extract_mountains(text)
        output.append({"text": text, "mountains": entities})
        if not args.json:
            print_result(text, entities, ner=ner, show_scores=args.show_scores)

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
