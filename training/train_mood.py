"""
Fine-tunes DistilBERT to classify a message into one of 28 GoEmotions labels.

    pip install -r requirements-train.txt
    python training/train_mood.py                 # full run
    python training/train_mood.py --quick         # small subset to check it works

Data:
- GoEmotions "simplified" (Google Research, ~58k Reddit comments). Only comments
  with exactly one label are kept, so this is plain single-label classification.
- training/custom_examples.csv: TunedIn-style messages you write yourself.
  They are repeated (--custom-weight) so a few hundred rows still matter.

Output: models/mood_classifier/ (loaded by src/mood_model.py).
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "models" / "mood_classifier"
QUICK_DIR = ROOT / "models" / "mood_classifier_quick"
CUSTOM_CSV = ROOT / "training" / "custom_examples.csv"
BASE_MODEL = "distilbert-base-uncased"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--custom-weight", type=int, default=5, help="times to repeat custom examples")
    parser.add_argument("--quick", action="store_true", help="train on 2,000 rows for a smoke test")
    args = parser.parse_args()

    import numpy as np
    from datasets import Dataset, concatenate_datasets, load_dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = raw["train"].features["labels"].feature.names  # 28 labels incl. neutral
    label2id = {n: i for i, n in enumerate(names)}

    def single_label(split):
        ds = raw[split].filter(lambda ex: len(ex["labels"]) == 1)
        return ds.map(lambda ex: {"label": ex["labels"][0]}).select_columns(["text", "label"])

    train, val, test = single_label("train"), single_label("validation"), single_label("test")

    custom = _load_custom(label2id)
    if custom:
        extra = Dataset.from_list(custom * args.custom_weight).cast(train.features)
        train = concatenate_datasets([train, extra]).shuffle(seed=42)
        print(f"Added {len(custom)} custom examples x{args.custom_weight}")

    if args.quick:
        train = train.shuffle(seed=42).select(range(2000))
        val = val.select(range(500))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok = lambda batch: tokenizer(batch["text"], truncation=True, max_length=128)
    train, val, test = (d.map(tok, batched=True) for d in (train, val, test))

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id=label2id,
    )

    def metrics(pred):
        preds = np.argmax(pred.predictions, axis=-1)
        return {"accuracy": float((preds == pred.label_ids).mean())}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(ROOT / "training" / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=64,
            learning_rate=5e-5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,
            logging_steps=100,
            report_to="none",
        ),
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()

    print("Held-out GoEmotions test set:", trainer.evaluate(test))
    # A --quick run is only a smoke test (~40% accuracy, mostly "neutral"), so it
    # saves elsewhere and never replaces the model the chatbot loads.
    out_dir = QUICK_DIR if args.quick else OUT_DIR
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved fine-tuned model to {out_dir}")
    if args.quick:
        print("Quick run: the chatbot will NOT use this model. Run without --quick for the real one.")


def _load_custom(label2id):
    if not CUSTOM_CSV.exists():
        return []
    rows = []
    with open(CUSTOM_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["emotion"] not in label2id:
                raise ValueError(f"Unknown emotion {row['emotion']!r} in {CUSTOM_CSV.name}")
            rows.append({"text": row["text"], "label": label2id[row["emotion"]]})
    return rows


if __name__ == "__main__":
    main()
