import os
import torch
import soundfile as sf
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import numpy as np

# Hugging Face repo ID
MODEL_REPO = "cactusZen/wav2vec2-hf"

# Emotion mapping from RAVDESS codes
ravdess_emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

print(f"Loading model from Hugging Face repo: {MODEL_REPO}")
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_REPO)
extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_REPO)
model.eval()

def predict(audio_path):
    waveform, sample_rate = sf.read(audio_path)

    # Ensure waveform is float32
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)

    if sample_rate != 16000:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        waveform = resampler(waveform).squeeze().numpy()

    inputs = extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        return model.config.id2label[predicted_id]

# Dataset path
DATA_DIR = "/Users/macintosh-computadora/Visual Studio Code/wav2vec-hf-workspace/Pytorch-Basics/data/ravdess"

file_names = []
y_true = []
y_pred = []

# Iterate over files
for root, _, files in os.walk(DATA_DIR):
    for file in tqdm(files, desc=f"Processing {root}"):
        if file.endswith(".wav"):
            true_emotion_code = file.split("-")[2]
            true_emotion = ravdess_emotion_map.get(true_emotion_code)

            if true_emotion is None:
                continue  # skip unknown codes

            audio_path = os.path.join(root, file)
            try:
                pred_emotion = predict(audio_path)
                file_names.append(file)       # store file name
                y_true.append(true_emotion)   # store true label
                y_pred.append(pred_emotion)   # store prediction
            except Exception as e:
                print(f"Error with {audio_path}: {e}")

# Save predictions
df = pd.DataFrame({"file": file_names, "true": y_true, "pred": y_pred})
df.to_csv("evaluation_results.csv", index=False)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
