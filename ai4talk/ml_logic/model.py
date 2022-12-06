import torch
import pandas as pd
from ai4talk.ml_logic.params import Config
from ai4talk.ml_logic.registry import WhisperModelModule, SpeechDataset, WhisperDataCollatorWhithPadding
import whisper
from tqdm.notebook import tqdm
from ai4talk.ml_logic.utils import highlight_diffs
def transcript(file_name):
    checkpoint_path = "/home/dhanya/code/aygul0790/ai4talk/checkpoints/checkpoint-epoch=0009.ckpt"
    state_dict = torch.load(checkpoint_path) #, map_location=torch.device('cpu'))
    state_dict = state_dict['state_dict']
    cfg = Config()
    whisper_model = WhisperModelModule(cfg)
    whisper_model.load_state_dict(state_dict)
    woptions = whisper.DecodingOptions(language="tt", without_timestamps=True) #, fp16 = False)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="tt", task=woptions.task)
    fpath = '/home/dhanya/code/aygul0790/ai4talk/notebooks/tat/' + str(file_name)
    df = pd.read_csv('/home/dhanya/code/aygul0790/ai4talk/notebooks/tat/metadata.csv')
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
 #   res = [highlight_diffs(x,y) for x,y in zip(refs,res)]
    return {"transcript true": refs, "transcript from model": res}

from ai4talk.ml_logic.utils import from_ipa_to_tat
def translate(data):
    data = from_ipa_to_tat(data)
    data = data.split()
    data = [x[1:] if x.startswith('йe') else x for x in data]
    data = [x.replace('йa', 'я') if x.startswith('йa') else x for x in data]
    data = [x.replace('йa', 'я') if 'йa' in x else x for x in data]
    data = [x.replace('ъ', 'ь') if x.endswith('ъ') else x for x in data]
    data = ' '.join(data)
    return data
