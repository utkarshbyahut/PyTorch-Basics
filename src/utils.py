# src/utils.py

EMOTION_LABELS = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7
}


"""
Extract emotion label from filename based on RAVDESS dataset naming convention.

def extract_label_from_filename(filename):
    #Parse emotion code from filename and return label.
    try:
        parts = filename.split("-")
        emotion_code = parts[2]
        return EMOTION_LABELS.get(emotion_code)
    except Exception as e:
        print(f"Failed to parse: {filename}")
        return None 
    
"""