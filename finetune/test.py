import os
import whisper

#model = whisper.load_model("large", download_root="./pre_models")

model = whisper.load_model("small", download_root="./pre_models")

# load audio and pad/trim it to fit 30 seconds

file_path = "tests/jfk.flac"

audio = whisper.load_audio(file_path)
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)


