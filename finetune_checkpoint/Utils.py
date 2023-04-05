import os

import torchaudio
import torch


def load_irish_data(file_path):
    audio_transcript_pair_list = []

    f = open(file_path)
    readlines = f.readlines()

    for line in readlines:
        line = line.rstrip("\n")
        data = line.split("\t")

        wav_file = data[0]
        wav_text = data[1]

        audio_id = os.path.basename(wav_file).split(".")[0]

        audio_transcript_pair_list.append((audio_id, str(wav_file), wav_text))

    return audio_transcript_pair_list


def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

