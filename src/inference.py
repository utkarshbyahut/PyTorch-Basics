import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Replace with your Hugging Face repo
MODEL_REPO = "cactusZen/wav2vec2-emotion"

# Load model + feature extractor from Hugging Face
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_REPO)
extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_REPO)

model.eval()

# Inference function
def predict(audio_path):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.squeeze()

    # Resample to 16kHz if needed
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

# Example
if __name__ == "__main__":
    test_audio = "test.wav"  # put a sample file here
    prediction = predict(test_audio)
    print(f"Predicted emotion: {prediction}")
