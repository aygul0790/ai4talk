import numpy as np
import pandas as pd
#from colored import fg
#olor = fg('blue')
#from termcolor import colored, cprint
import difflib as dl
def highlight_diffs(str1, str2):
  """
  prints the second string with bold and red cHaracter where it is different from 1st string
  """
  result = ""
  for i in range(min(len(str1), len(str2))):
    if str1[i] == str2[i]:
      result += str1[i]
    else:
       result += '<span class="special">'+str2[i]+'</span>'
      #result += f"__{str2[i]}__"
    #result +=str2[i]
  #result += str2[i+1:]
  return result
