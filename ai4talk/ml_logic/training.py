import torch

import numpy as np
import pandas as pd

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ai4talk.ml_logic.params import Config
from ai4talk.ml_logic.registry import WhisperModelModule, SpeechDataset, WhisperDataCollatorWhithPadding
import whisper

from tqdm.notebook import tqdm

from pathlib import Path
import evaluate

# try:
#     from collections.abc import Iterable
# except ImportError:
#     from collections import Iterable

# from abydos import distance

from ai4talk.ml_logic.utils import highlight_diffs

####################################################################################################
TRAIN_RATE = 0.8
SEED = 42
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
seed_everything(SEED, workers=True)

####################################################################################################

df = pd.read_csv('/home/aygul_unix/code/aygul0790/ai4talk/notebooks/new_audio/tat/metadata.csv')
df['audio_path'] = '/home/aygul_unix/code/aygul0790/ai4talk/notebooks/new_audio/tat/' + df['file_name']
df['text'] = df['transcription']
df = df.reset_index()
df['audio_id'] = df['index']

audio_transcript_pair_list = list(df[['audio_id', 'audio_path', 'text']].itertuples(index=False, name=None))

train_num = int(len(audio_transcript_pair_list) * TRAIN_RATE)
train_audio_transcript_pair_list, eval_audio_transcript_pair_list = audio_transcript_pair_list[:train_num], audio_transcript_pair_list[train_num:]

####################################################################################################

woptions = whisper.DecodingOptions(language="tt", without_timestamps=True) # tatar language
wmodel = whisper.load_model("base")
wtokenizer = whisper.tokenizer.get_tokenizer(True, language="tt", task=woptions.task)

################################## Tensorboard related #############################################

log_output_dir = "/home/aygul_unix/code/aygul0790/ai4talk/logs"
check_output_dir = "/home/aygul_unix/code/aygul0790/ai4talk/checkpoints"

train_name = "whisper"
train_id = "00001"

model_name = "base"
lang = "tt"

######################################################################################################

checkpoint_path = "/home/aygul_unix/code/aygul0790/ai4talk/checkpoints/checkpoint-epoch=0009.ckpt"

cfg = Config()

Path(log_output_dir).mkdir(exist_ok=True)
Path(check_output_dir).mkdir(exist_ok=True)

tflogger = TensorBoardLogger(
    save_dir=log_output_dir,
    name=train_name,
    version=train_id
)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{check_output_dir}/checkpoint",
    filename="checkpoint-{epoch:04d}",
    save_top_k=-1 # all model save
)

callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

model = WhisperModelModule(cfg, model_name, lang, train_audio_transcript_pair_list, eval_audio_transcript_pair_list)

trainer = Trainer(
    precision=16,
    accelerator=DEVICE,
    max_epochs=cfg.num_train_epochs,
    accumulate_grad_batches=cfg.gradient_accumulation_steps,
    logger=tflogger,
    callbacks=callback_list
)

# trainer.fit(model, ckpt_path=checkpoint_path)

trainer.fit(model)


######################## Check the performance of the model ############################################

last_checkpoint_path = "/home/aygul_unix/code/aygul0790/ai4talk/checkpoints/checkpoint/checkpoint-epoch=0016.ckpt"

state_dict = torch.load(last_checkpoint_path)
# print(state_dict.keys())
state_dict = state_dict['state_dict']

whisper_model = WhisperModelModule(cfg)
whisper_model.load_state_dict(state_dict)

woptions = whisper.DecodingOptions(language="tt", without_timestamps=True)
dataset = SpeechDataset(eval_audio_transcript_pair_list, wtokenizer, 16000)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())

refs = []
res = []
for b in tqdm(loader):
    input_ids = b["input_ids"].half().cuda()
    labels = b["labels"].long().cuda()
    with torch.no_grad():
        #audio_features = whisper_model.model.encoder(input_ids)
        #out = whisper_model.model.decoder(enc_input_ids, audio_features)
        results = whisper_model.model.decode(input_ids, woptions)
        for r in results:
            res.append(r.text)

        for l in labels:
            l[l == -100] = wtokenizer.eot
            ref = wtokenizer.decode(l, skip_special_tokens=True)
            refs.append(ref)


cer_metrics = evaluate.load("cer")
print(cer_metrics.compute(references=refs, predictions=res))

wer_metrics = evaluate.load("wer")
print(wer_metrics.compute(references=refs, predictions=res))


# phonetic = distance.PhoneticEditDistance()

# ped_metrics_max = max([phonetic.dist(l, o) for l,o in zip(refs,res)])
# ped_metrics_mean = np.mean([phonetic.dist(l, o) for l,o in zip(refs,res)])
# print(ped_metrics_max)
# print()
# print(ped_metrics_mean)


# for k, v in zip(refs, res):
#     print("-"*10)
#     print(k)
#     print()
#     print(highlight_diffs(k,v))
