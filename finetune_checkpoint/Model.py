
import torch
import whisper
import evaluate

from torch import nn

from pytorch_lightning import LightningModule
from transformers import (AdamW, get_linear_schedule_with_warmup)

from finetune_checkpoint.Dataset import IrishSpeechDataset
from finetune_checkpoint.DataCollator import WhisperDataCollatorWhithPadding


class WhisperModelModule(LightningModule):
    def __init__(self, config, train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        
        self.options = whisper.DecodingOptions(language=config["tokenizer_language"], without_timestamps=True)
        self.model = whisper.load_model(config["model_size"])
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=config["tokenizer_language"], task=self.options.task)

        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.config = config
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            #o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            o_list.append(self.tokenizer.decode(o))
            #l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
            l_list.append(self.tokenizer.decode(l))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.config["learning_rate"],
                          eps=self.config["adam_epsilon"],
                          no_deprecation_warning=True)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.config["batch_size"]))
                // self.config["gradient_accumulation_steps"]
                * float(self.config["num_train_epochs"])
            )

    def train_dataloader(self):
        dataset = IrishSpeechDataset(self.__train_dataset, self.tokenizer, self.config["sample_rate"])
        return torch.utils.data.DataLoader(dataset,
                          batch_size=self.config["batch_size"],
                          drop_last=True, shuffle=True, num_workers=self.config["num_worker"],
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        dataset = IrishSpeechDataset(self.__eval_dataset, self.tokenizer, self.config["sample_rate"])
        return torch.utils.data.DataLoader(dataset,
                          batch_size=self.config["batch_size"],
                          num_workers=self.config["num_worker"],
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

