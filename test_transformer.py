import torch
import torchaudio

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Load model
model = AutoModelForSpeechSeq2Seq.from_pretrained("whisper_finetune_small_English").to(device)
processor = AutoProcessor.from_pretrained("whisper_finetune_small_English", language="None", task="transcribe")

# NB: set forced_decoder_ids for generation utils
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="de", task="transcribe")

# 16_000
model_sample_rate = processor.feature_extractor.sampling_rate

# Load data
ds_mcv_test = load_dataset("mozilla-foundation/common_voice_11_0", "de", split="test", streaming=True)
test_segment = next(iter(ds_mcv_test))
waveform = torch.from_numpy(test_segment["audio"]["array"])
sample_rate = test_segment["audio"]["sampling_rate"]

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

# Normalise predicted sentences if necessary

