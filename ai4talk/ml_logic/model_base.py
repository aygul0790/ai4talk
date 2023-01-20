import os
import torch
import torchaudio

import pandas as pd
from transformers import AutoModelForCTC, Wav2Vec2Processor


def transcript(file_name):
    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

    device_s = f"cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_s)

    model = model.to(device)

    fpath = '/home/aygul_unix/code/aygul0790/ai4talk/notebooks/new_audio/tat/' + str(file_name)

    waveform, sample_rate = torchaudio.load(fpath)
    waveform = waveform.to(device)
    logits = model(waveform).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred_str = processor.batch_decode(pred_ids)[0]

    return pred_str

# This part is used if there are errors related to GPU not being found!

# def transcript(file_name):

#     model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
#     processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

#     #device_s = "cpu"
#     #device = torch.device(device_s)

#     fpath = '/home/aygul_unix/code/aygul0790/ai4talk/notebooks/new_audio/tat/' + str(file_name)

#     waveform, sample_rate = torchaudio.load(fpath)
#     waveform = waveform
#     logits = model(waveform).logits
#     pred_ids = torch.argmax(logits, dim=-1)
#     pred_str = processor.batch_decode(pred_ids)[0]

#     return {"transcription": pred_str}
