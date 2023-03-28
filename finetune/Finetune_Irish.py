import os


from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor



class FinetuneIrish(object):
    def __init__(self, config):
        
        feature_extractor, tokenizer, processor = self.load_whisper_resource(config)
        
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.processor = processor


    def load_whisper_resource(self, config):
        
        str_openai = "openai/whisper-" + config["model_size"]

        feature_extractor = WhisperFeatureExtractor.from_pretrained(str_openai)
        tokenizer = WhisperTokenizer.from_pretrained(str_openai, language=config["tokenizer_language"], task="transcribe")
        processor = WhisperProcessor.from_pretrained(str_openai, language=config["processor_language"], task="transcribe")
        
        return feature_extractor, tokenizer, processor


