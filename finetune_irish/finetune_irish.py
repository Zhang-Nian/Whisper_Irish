import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import resampy
import soundfile as sf
import numpy as np

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="None", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


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
        cur_audio["sampling_rate"] = 16000
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

#print(irish_common_voice["train"][0])


class IrishDataset(Dataset):
    def __init__(self, data_list):
        
        self.data_list = data_list
    
    def __getitem__(self, index):
    
        cur_data = self.data_list[index]

        audio = cur_data["audio"]

        cur_data["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        cur_data["labels"] = tokenizer(cur_data["sentence"]).input_ids

        return cur_data

    
    def __len__(self):
        return len(self.data_list)


train_dataset = IrishDataset(irish_common_voice["train"])
test_dataset = IrishDataset(irish_common_voice["test"])

print("generate test finish !!")



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# Define the Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./irish-small",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=20000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)


print("irish training finish")

