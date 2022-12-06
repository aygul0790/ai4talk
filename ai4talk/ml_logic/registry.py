import numpy as np
import torch
from torch import nn
import torchaudio.transforms as at
import whisper
from pytorch_lightning import LightningModule
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
import evaluate
# import Config params
from ai4talk.ml_logic.params import Config
# import PED
# from abydos import distance
# phonetic = distance.PhoneticEditDistance()
# import load_wave
from ai4talk.ml_logic.utils import load_wave
#############################################################################################################################
class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate) -> None:
        super().__init__()
        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.audio_info_list)
    def __getitem__(self, id):
        audio_id, audio_path, text = self.audio_info_list[id]
        # audio
        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text
        }

###########################################################################################################
class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id
        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }
        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids
        return batch
class WhisperModelModule(LightningModule):
    def __init__(self, cfg:Config, model_name="base", lang="tt", train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="tt", task=self.options.task)
        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")
      #  self.metrics_ped = phonetic
        self.cfg = cfg
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
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)
      #  ped = max([self.metrics_ped.dist(l, o) for l,o in zip(l_list,o_list)])
        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)
      #  self.log("val/ped", ped, on_step=True, prog_bar=True, logger=True)
        return {
          #  "ped": ped,
            "cer": cer,
            "wer": wer,
            "loss": loss
        }
    def configure_optimizers(self):
        """ Whisper optimizer """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
# optimizer step
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.cfg.learning_rate,
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer
# scheduler step
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler
# there will be a warning!
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    def setup(self, stage=None):
        """  """
        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
    def train_dataloader(self):
        """  """
        dataset = SpeechDataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset,
                          batch_size=self.cfg.batch_size,
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )
    def val_dataloader(self):
        """  """
        dataset = SpeechDataset(self.__eval_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset,
                          batch_size=self.cfg.batch_size,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )
