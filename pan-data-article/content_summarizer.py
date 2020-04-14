from summarizer import Summarizer
from nltk.tokenize import sent_tokenize
import os

model = Summarizer()
files = ['train1.txt', 'train2.txt', 'valid1.txt', 'valid2.txt']

for filename in files:
  filepath = f'content/{filename}'
  f_sum = open(f'summary/{filename}', 'w+')
  if os.path.exists(filepath):
    print(filepath)
    with open(filepath, 'r') as f:
      i = 0
      for content in f:
        try:
          i += 1
          if i%10 == 0:
            print(i)
          result = model(content, ratio=0.2, max_length=200)
          sentences = sent_tokenize(result)
          for s in sentences:
            f_sum.write(f'{s}\n')
        except ValueError:
          continue

  f_sum.close()
