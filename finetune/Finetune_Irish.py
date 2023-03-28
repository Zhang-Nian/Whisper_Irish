import os
import evaluate

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer


class FinetuneIrish(object):
    def __init__(self, config):
        
        feature_extractor, tokenizer, processor, model = self.load_whisper_resource(config)
        
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model

        self.metric = evaluate.load("wer")

        # Define the Training Arguments
        self.training_args = self.set_training_arguments(config)


    def load_whisper_resource(self, config):
        
        str_openai = "openai/whisper-" + config["model_size"]

        feature_extractor = WhisperFeatureExtractor.from_pretrained(str_openai)
        tokenizer = WhisperTokenizer.from_pretrained(str_openai, language=config["tokenizer_language"], task="transcribe")
        processor = WhisperProcessor.from_pretrained(str_openai, language=config["processor_language"], task="transcribe")
        
        model = WhisperForConditionalGeneration.from_pretrained(str_openai)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        return feature_extractor, tokenizer, processor, model

    
    def set_training_arguments(self, config):
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=config["outputdir"],  # change to a repo name of your choice
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],  # increase by 2x for every 2x decrease in batch size
            learning_rate=config["learning_rate"],
            warmup_steps=config["warmup_steps"],
            max_steps=config["max_steps"],
            gradient_checkpointing=config["gradient_checkpointing"],
            fp16=config["fp16"],
            evaluation_strategy=config["evaluation_strategy"],
            per_device_eval_batch_size=config["per_device_eval_batch_size"],
            predict_with_generate=config["predict_with_generate"],
            generation_max_length=config["generation_max_length"],
            save_steps=config["save_steps"],
            eval_steps=config["eval_steps"],
            logging_steps=config["logging_steps"],
            load_best_model_at_end=config["load_best_model_at_end"],
            metric_for_best_model=config["metric_for_best_model"],
            greater_is_better=config["greater_is_better"]
        )

        return training_args

    def train(self, config, train_dataset, test_dataset, data_collator):
        
        trainer = Seq2SeqTrainer(
            args=self.training_args,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        trainer.train()

        # save result
        self.model.save_pretrained(self.training_args.output_dir)
        self.processor.save_pretrained(self.training_args.output_dir)


    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

