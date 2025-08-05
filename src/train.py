# src/train.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", message=".*LibreSSL*", category=UserWarning)

import os
import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
from utils import EMOTION_LABELS
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Constants
DATA_DIR = "data/ravdess"
CSV_PATH = os.path.join(DATA_DIR, "ravdess.csv")

def preprocess(batch):
    audio_path = batch["path"]
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.squeeze()

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    inputs = extractor(
        waveform.numpy(),
        sampling_rate=16000,
        padding="max_length",
        truncation=True,
        max_length=16000 * 4,
        return_tensors="np"
    )

    batch["input_values"] = inputs["input_values"][0]
    batch["attention_mask"] = inputs["attention_mask"][0]
    batch["label"] = int(EMOTION_LABELS[batch["emotion"]])
    return batch

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    labels = pred.label_ids
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def main():
    # 1. Load Dataset
    ds = load_dataset("csv", data_files=CSV_PATH, split="train")

    # 2. Split into train and test
    ds = ds.train_test_split(test_size=0.2, seed=42)
    train_ds, test_ds = ds["train"], ds["test"]

    # 3. Load Feature Extractor
    global extractor
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base",
        return_attention_mask=True
    )

    # 4. Preprocess Dataset
    train_ds = train_ds.map(preprocess)
    test_ds  = test_ds.map(preprocess)

    # 5. Load Model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=len(EMOTION_LABELS),
        label2id=EMOTION_LABELS,
        id2label={v: k for k, v in EMOTION_LABELS.items()}
    )

    # 6. Training Arguments
    args = TrainingArguments(
        output_dir="models",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        fp16=torch.cuda.is_available(),
        learning_rate=1e-4,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=extractor,
        compute_metrics=compute_metrics,
    )

    # 8. Train and Save
    trainer.train()
    trainer.save_model("models/wav2vec2-emotion")
    extractor.save_pretrained("models/wav2vec2-emotion")

if __name__ == "__main__":
    main()
