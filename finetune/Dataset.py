import os


from torch.utils.data import Dataset

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer


class IrishDataSet(Dataset):
    def __init__(self, config, data_list):

        feature_extractor, tokenizer = self.load_whisper_resource(config)

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        self.data_list = data_list


    def load_whisper_resource(self, config):

        str_openai = "openai/whisper-" + config["model_size"]
        
        feature_extractor = WhisperFeatureExtractor.from_pretrained(str_openai)
        tokenizer = WhisperTokenizer.from_pretrained(str_openai, language=config["tokenizer_language"], task="transcribe")

        return feature_extractor, tokenizer

    def __getitem__(self, index):
    
        cur_data = self.data_list[index]

        audio = cur_data["audio"]

        cur_data["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        cur_data["labels"] = self.tokenizer(cur_data["sentence"]).input_ids

        return cur_data

    
    def __len__(self):
        return len(self.data_list)

