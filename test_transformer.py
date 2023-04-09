import soundfile as sf
import numpy as np

import torch
import torchaudio

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Load model
model = AutoModelForSpeechSeq2Seq.from_pretrained("./whisper_finetune_small_English").to(device)
processor = AutoProcessor.from_pretrained("./whisper_finetune_small_English", language="None", task="transcribe")

# NB: set forced_decoder_ids for generation utils
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="de", task="transcribe")

# 16_000
model_sample_rate = processor.feature_extractor.sampling_rate

# Load data

wav_path = "/media/storage/phonetics/asr_data_irish/recognition/mileglor/oir22_extraspks/06cc6562-0562-4efa-b620-81b6d5b3de33_grownups_0072/06cc6562-0562-4efa-b620-81b6d5b3de33_grownups_0072.wav"

audio, rate = sf.read(wav_path)

#waveform = torch.from_numpy(audio)
waveform = torch.from_numpy(audio.astype(np.float32))
sample_rate = rate

# Resample
if sample_rate != model_sample_rate:
    resampler = torchaudio.transforms.Resample(sample_rate, model_sample_rate)
    waveform = resampler(waveform)

# Get feat
inputs = processor(waveform, sampling_rate=model_sample_rate, return_tensors="pt")
input_features = inputs.input_features
input_features = input_features.to(device)

# Generate
generated_ids = model.generate(inputs=input_features, max_new_tokens=225)  # greedy
# generated_ids = model.generate(inputs=input_features, max_new_tokens=225, num_beams=5)  # beam search

# Detokenize
generated_sentences = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("generated_sentences is :", generated_sentences)

# Normalise predicted sentences if necessary

