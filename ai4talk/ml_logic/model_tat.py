import torch

import pandas as pd
import os

from ai4talk.ml_logic.params import Config
from ai4talk.ml_logic.registry import WhisperModelModule, SpeechDataset, WhisperDataCollatorWhithPadding
import whisper

from tqdm.notebook import tqdm


def transcript_tat(file_name):

    checkpoint_path =  os.path.join(os.getcwd(),"checkpoints", "checkpoint_tat.ckpt")

    state_dict = torch.load(checkpoint_path) #, map_location=torch.device('cpu'))
    state_dict = state_dict['state_dict']

    cfg = Config()

    whisper_model = WhisperModelModule(cfg)
    whisper_model.load_state_dict(state_dict)

    woptions = whisper.DecodingOptions(language="tt", without_timestamps=True) #, fp16 = False)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="tt", task=woptions.task)

    fpath = os.path.join(os.getcwd(),"processed_data","tat", file_name)
    df = pd.read_csv(os.path.join(os.getcwd(),"processed_data","tat", "metadata.csv"))
    df['audio_path'] = fpath
    df['text'] = df['transcription']
    df = df.reset_index()
    df['audio_id'] = df['index']
    df = df[df['file_name'] == file_name]

    eval_audio_transcript_pair_list = list(df[['audio_id', 'audio_path', 'text']].itertuples(index=False, name=None))

    dataset = SpeechDataset(eval_audio_transcript_pair_list, wtokenizer, 16000)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=WhisperDataCollatorWhithPadding())

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

    return {"transcript true": refs, "transcript from model": res}
