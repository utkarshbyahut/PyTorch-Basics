import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import soundfile as sf

# Your Hugging Face repo
MODEL_REPO = "cactusZen/wav2vec2-hf"

# Load model + feature extractor
print(f"Loading model from Hugging Face repo: {MODEL_REPO}")
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_REPO)
extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_REPO)

model.eval()



def predict(audio_path):
    waveform, sample_rate = sf.read(audio_path)
    waveform = torch.tensor(waveform, dtype=torch.float32)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Extract features
    inputs = extractor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    # Forward pass
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()

    # Map ID to label
    label = model.config.id2label[predicted_id]
    return label

if __name__ == "__main__":
    # Pick any RAVDESS file from your dataset
    test_audio = "/Users/macintosh-computadora/Visual Studio Code/wav2vec-hf-workspace/Pytorch-Basics/data/ravdess/Actor_01/03-01-01-01-01-01-01.wav"

    prediction = predict(test_audio)
    print(f"Predicted emotion for {test_audio}: {prediction}")
