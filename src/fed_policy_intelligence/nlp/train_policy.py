"""Fine-tune an accessible RoBERTa checkpoint on expert FOMC stance labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from fed_policy_intelligence.data.expert_labels import LABEL_NAMES, load_expert_dataset
from fed_policy_intelligence.evaluation.classification import evaluate_labels

BASE_MODEL_ID = "FacebookAI/roberta-base"
BASE_MODEL_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"


def _metric_payload(expected: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(logits).argmax(axis=1)
    report = evaluate_labels(
        [LABEL_NAMES[int(value)] for value in expected],
        [LABEL_NAMES[int(value)] for value in predicted],
        labels=[LABEL_NAMES[index] for index in sorted(LABEL_NAMES)],
    )
    accuracy = float(np.mean(np.asarray(expected) == predicted))
    return {"accuracy": accuracy, "macro_f1": report.macro_f1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-cache-dir", default="cache/huggingface/datasets")
    parser.add_argument("--model-cache-dir", default="cache/huggingface/models")
    parser.add_argument("--output-dir", default="artifacts/models/fomc-roberta-base")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive.")
    if not 0 < args.validation_size < 0.5:
        parser.error("--validation-size must be between 0 and 0.5.")

    try:
        import torch
        from datasets import Dataset
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with: "
            "pip install -e '.[nlp,training]'"
        ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    expert = load_expert_dataset(cache_dir=args.dataset_cache_dir)
    train_indices, validation_indices = train_test_split(
        np.arange(len(expert.train)),
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=expert.train["label_id"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        cache_dir=args.model_cache_dir,
    )

    def build_dataset(indices: np.ndarray) -> Any:
        frame = expert.train.iloc[indices][["sentence", "label_id"]].rename(
            columns={"label_id": "labels"}
        )
        dataset = Dataset.from_pandas(frame, preserve_index=False)
        return dataset.map(
            lambda batch: tokenizer(batch["sentence"], truncation=True, max_length=512),
            batched=True,
            remove_columns=["sentence"],
        )

    training_data = build_dataset(train_indices)
    validation_data = build_dataset(validation_indices)
    output_dir = Path(args.output_dir).resolve()
    training_dir = output_dir.parent / f".{output_dir.name}-training"
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        cache_dir=args.model_cache_dir,
        num_labels=3,
        id2label=LABEL_NAMES,
        label2id={label: index for index, label in LABEL_NAMES.items()},
    )
    training_args = TrainingArguments(
        output_dir=str(training_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=args.seed,
        data_seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=validation_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=lambda result: _metric_payload(result.label_ids, result.predictions),
    )
    train_result = trainer.train()
    validation_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    provenance = {
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "dataset_id": expert.dataset_id,
        "dataset_revision": expert.revision,
        "training_rows": len(training_data),
        "validation_rows": len(validation_data),
        "untouched_test_rows": len(expert.test),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
    }
    (output_dir / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({**provenance, "output_dir": str(output_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
