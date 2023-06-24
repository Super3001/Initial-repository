'''from datasets import load_from_disk
print('hello')
ds = load_from_disk('./dataset')
ds = ds['train']

import pandas as pd

df = pd.DataFrame(ds)
df_100 = df.iloc[:100,:]
df_100.to_excel('dataset_100.xlsx')'''

import torch
from datasets import load_dataset
ds = load_dataset("go_emotions", "raw", split='train[0:100]')

import pandas as pd

emotions = [
 'admiration',
 'amusement',
 'anger',
 'annoyance',
 'approval',
 'caring',
 'confusion',
 'curiosity',
 'desire',
 'disappointment',
 'disapproval',
 'disgust',
 'embarrassment',
 'excitement',
 'fear',
 'gratitude',
 'grief',
 'joy',
 'love',
 'nervousness',
 'optimism',
 'pride',
 'realization',
 'relief',
 'remorse',
 'sadness',
 'surprise',
 'neutral']

ds = ds.map(lambda x: {"labels": torch.tensor([torch.tensor(x[c]) for c in emotions]).to(torch.long)})

# ds = ds.map(lambda x: {"emotion": emotions[int(torch.argmax(x["labels"]))]})
cols = ds.column_names
cols.remove('text')
cols.remove('example_very_unclear')
cols.remove('labels')
ds = ds.map(lambda x: {"done": 1}, remove_columns=cols)

df = pd.DataFrame(ds)

import numpy as np
def translate(x):
    return emotions[int(np.argmax(x['labels']))]

df['emotion'] = df.apply(translate, axis=1)

df = df.drop('labels', axis=1)

df.to_excel('raw dataset_100.xlsx')

