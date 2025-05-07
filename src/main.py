import evaluate
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import numpy as np
from loguru import logger
import polars as pl
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, SEED, TRAIN_SPLIT, VALIDATION_SPLIT

app = typer.Typer()

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6).to(device))
metric = evaluate.load("accuracy")


def tokenize(examples):
    outputs = tokenizer(examples["text"], truncation=True)
    return outputs


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


@app.command()
def main():
    logger.info("Downloading dataset...")
    df = pl.read_parquet("hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet")
    logger.success("Downloading dataset complete.")

    logger.info("Processing dataset...")
    df = df.with_columns(
        [
            pl.col("text").str.replace(r"(\n|\r|\s+)", " ").str.strip_chars().str.normalize("NFKD"),
        ]
    )

    df = df.filter(pl.col("text").is_not_null() & pl.col("label").is_not_null())
    df = df.filter(pl.col("text") != "")
    df = df.unique(subset=["text", "label"])
    df = df.rename({"label": "labels"})
    logger.success("Processing dataset complete.")

    logger.info("Splitting dataset...")
    n_samples = df.height
    train_size = int(n_samples * TRAIN_SPLIT)
    validation_size = int(n_samples * VALIDATION_SPLIT)
    logger.info(
        f"Train size: {train_size}, Validation size: {validation_size}, Test size: {n_samples - train_size - validation_size}"
    )

    train_df = df.sample(n=train_size, with_replacement=False, seed=SEED)
    remaining_df = df.join(train_df, on="text", how="anti")
    validation_df = remaining_df.sample(n=validation_size, with_replacement=False, seed=SEED)
    test_df = remaining_df.join(validation_df, on="text", how="anti")
    logger.success("Splitting dataset complete.")

    logger.info("Saving dataset splits...")
    train_df.write_parquet(PROCESSED_DATA_DIR / "train.parquet")
    validation_df.write_parquet(PROCESSED_DATA_DIR / "validation.parquet")
    test_df.write_parquet(PROCESSED_DATA_DIR / "test.parquet")

    ds = load_dataset(
        "parquet",
        data_files={
            "train": str(PROCESSED_DATA_DIR / "train.parquet"),
            "validation": str(PROCESSED_DATA_DIR / "validation.parquet"),
        },
    )

    logger.info("Starting training...")
    tokenized_ds = ds.map(tokenize, batched=True, batch_size=None)
    tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        num_train_epochs=1,
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        output_dir=MODELS_DIR / "distilbert-twitter-emotions-checkpoint",
        push_to_hub=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    logger.success("Training complete.")

    logger.info("Evaluating model...")
    test_ds = load_dataset(
        "parquet",
        data_files={
            "test": str(PROCESSED_DATA_DIR / "test.parquet"),
        },
    )
    test_ds = test_ds.map(tokenize, batched=True)

    test_results = trainer.evaluate(test_ds["test"])
    logger.success("Evaluation complete.")
    logger.info(f"Test results: {test_results}")

if __name__ == "__main__":
    app()
