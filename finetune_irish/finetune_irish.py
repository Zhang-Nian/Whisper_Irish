import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import resampy
import soundfile as sf
import numpy as np

from sklearn.model_selection import train_test_split

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="None", task="transcribe")


def load_irish_data():
    wav_list = []
    text_list = []

    file_path = "train.list"
    f = open(file_path)
    readlines = f.readlines()

    for line in readlines:
        line = line.rstrip("\n")
        data = line.split("\t")

        wav_file = data[0]
        wav_text = data[1]

        wav_list.append(wav_file)
        text_list.append(wav_text)

    return wav_list, text_list


def generate_list(wav, text):

    assert(len(wav) == len(text))
    
    final_list = []

    for i in range(len(wav)):
        cur_wav = wav[i]
        cur_text = text[i]
        
        cur_map = {}
        cur_audio = {}
        cur_audio["path"] = cur_wav

        # read wav
        audio, rate = sf.read(cur_wav)
        if rate != 16000:
            audio = resampy.resample(audio.astype(np.float32), rate, 16000, axis=0)

        cur_audio["array"] = audio
        cur_map["audio"] = cur_audio
        cur_map["sampling_rate"] = 16000
        cur_map["sentence"] = cur_text
        final_list.append(cur_map)

    return final_list


def make_special_format_like_commonvoice(wav_train, wav_test, text_train, text_test):
    common_voice = {}

    # process train 
    train_list = generate_list(wav_train, text_train)
    common_voice["train"] = train_list

    # process test
    test_list = generate_list(wav_test, text_test)
    common_voice["test"] = test_list

    return common_voice


# load irish speech and split into train and test dataset
wav_list, text_list = load_irish_data()
train_wav, test_wav, train_text, test_text = train_test_split(wav_list, text_list, test_size=0.2)

print("train size is :", len(train_text))
print("test size is :", len(test_wav))

# generate special format in order to train
irish_common_voice = make_special_format_like_commonvoice(train_wav, test_wav, train_text, test_text)

print(irish_common_voice["train"][0])



