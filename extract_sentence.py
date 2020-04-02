import pandas as pd
import os

dataset_multifc = '~/Documents/UiS/Thesis/dataset/public_data_multi_fc'
test = pd.read_csv(f'{dataset_multifc}/test.tsv', sep='\t', header=None)

for id in test[0].iteritems():
  filepath = f"../dataset/public_data_multi_fc/snippets/{id[1]}"
  if (os.path.exists(filepath)):
    with open(filepath, 'r') as f:
      content = f.read()
      print(content)
  break