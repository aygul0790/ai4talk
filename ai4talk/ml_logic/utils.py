import numpy as np

import torch
import torchaudio
import torchaudio.transforms as at


def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


def highlight_diffs(str1, str2):
  """
  prints the second string with bold and red caracter where it is different from 1st string
  """
  result = ""
  for i in range(min(len(str1), len(str2))):
    if str1[i] == str2[i]:
      result += str1[i]
    else:
      result += f"\033[1;31m{str2[i]}\033[00m"
  result += str2[i+1:]
  return result


mapping = { # consonants

            'б' : 'b',
            'р' : 'r',
            'п' : 'p',
            'н' : 'n',
            'с' : 's',
            'т' : 't',
            'л' : 'l',
            'ф' : 'f',
            'к' : 'k',
            'й' : 'j',
            'д' : 'd',
            'м' : 'm',
            'ч' : 'ɕ',
            'ш' : 'ʃ',
            'ц' : 'ts',
            'г' : 'ɡ',
            'з' : 'z',
            'в' : 'w',
            'җ' : 'ʑ',
            'һ' : 'h',
            'ң' : 'ŋ',
            'х' : 'x',
            'ж' : 'ʒ',
            'ь' : 'ʔ',

            #'щ' : ???, <-- seems like it doesn't exist and also in Wiki it's written that it is only used in Russian words
            'ъ' : 'ʔ', # <-- ambigious, the same for both 'ъ' and 'ь'
            ' ' : ' ',

           # vowels
           'e' : 'e',
           'и' : 'i',
           'о' : 'o',
           'у' : 'u',
           'ы' : 'ɤ',
           'a' : 'ɑ',
           'ө' : 'ø',
           'ү' : 'y',
           'ә' : 'a',
           'ю' : 'ɯ'}

inverse_mapping = {value: key for key, value in mapping.items()}

def from_ipa_to_tat(string):
    """
    function that takes a string in phonetic alphabet and returns a string (without any space) in tatar alphabet
    """
    new_string= ''

    for letter in string:
        if letter in inverse_mapping.keys():

            new_string+= inverse_mapping[letter]

    return new_string

def translate(data):

    data = from_ipa_to_tat(data)
    data = data.split()

    data = [x[1:] if x.startswith('йe') else x for x in data]
    data = [x.replace('йa', 'я') if x.startswith('йa') else x for x in data]
    data = [x.replace('йa', 'я') if 'йa' in x else x for x in data]
    data = [x.replace('ъ', 'ь') if x.endswith('ъ') else x for x in data]

    data = ' '.join(data)

    return data
