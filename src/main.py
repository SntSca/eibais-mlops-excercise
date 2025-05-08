from datasets import load_dataset
import evaluate
from loguru import logger
import numpy as np
import polars as pl
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import typer

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)

app = typer.Typer()

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    df = pl.read_parquet(
        "hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet"
    )
    logger.success("Downloading dataset complete.")

    logger.info("Processing dataset...")
    df = df.with_columns(
        [
            pl.col("text")
            .str.replace(r"(\n|\r|\s+)", " ")
            .str.strip_chars()
            .str.normalize("NFKD"),
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
    validation_df = remaining_df.sample(
        n=validation_size, with_replacement=False, seed=SEED
    )
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
    training_ds = (
        ds["train"].shuffle(seed=SEED).select(range(1000))
    )  # Selecting a subset for faster training
    training_ds = training_ds.map(tokenize, batched=True, batch_size=None)
    training_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    validation_ds = (
        ds["validation"].shuffle(seed=SEED).select(range(500))
    )  # Selecting a subset for faster evaluation
    validation_ds = validation_ds.map(tokenize, batched=True, batch_size=None)
    validation_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id2label = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise",
    }
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=6, id2label=id2label, label2id=label2id
    ).to(device)
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=training_ds,
        eval_dataset=validation_ds,
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
    test_ds = (
        test_ds["test"].shuffle(SEED).select(range(100)).map(tokenize, batched=True)
    )  # Selecting a subset for faster evaluation

    test_results = trainer.evaluate(test_ds)
    logger.success("Evaluation complete.")
    logger.info(f"Test results: {test_results}")


if __name__ == "__main__":
    app()
