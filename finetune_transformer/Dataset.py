import os
import resampy
import numpy as np
import soundfile as sf

from torch.utils.data import Dataset

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer


class IrishDataSet(Dataset):
    def __init__(self, config, wav_list, text_list):

        feature_extractor, tokenizer = self.load_whisper_resource(config)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        self.wav_list = wav_list
        self.text_list = text_list


    def load_whisper_resource(self, config):

        str_openai = "openai/whisper-" + config["model_size"]
        
        feature_extractor = WhisperFeatureExtractor.from_pretrained(str_openai)
        tokenizer = WhisperTokenizer.from_pretrained(str_openai, language=config["tokenizer_language"], task="transcribe")

        return feature_extractor, tokenizer

    def __getitem__(self, index):

        cur_wav = self.wav_list[index]
        cur_text = self.text_list[index]

        audio, rate = sf.read(cur_wav)
        if rate != 16000:
            audio = resampy.resample(audio.astype(np.float32), rate, 16000, axis=0)

        cur_data = {}
        cur_data["input_features"] = self.feature_extractor(audio, sampling_rate=16000).input_features[0]
        cur_data["labels"] = self.tokenizer(cur_text).input_ids

        return cur_data

    
    def __len__(self):
        return len(self.wav_list)

