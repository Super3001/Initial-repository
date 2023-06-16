import torch
from datasets import load_dataset
ds = load_dataset("go_emotions", "raw")
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
model_name = "microsoft/xtremedistil-l6-h384-uncased"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
ds = ds.map(lambda x: {"labels": [x[c] for c in emotions]})

def tokenize_function(ds):
    return tokenizer(ds["text"], padding="max_length", truncation=True, max_length=64)

cols = ds["train"].column_names
cols.remove("labels")
ds_enc = ds.map(tokenize_function, batched=True, remove_columns=cols)
print(ds_enc)
ds_enc.set_format("torch")
ds_enc = (ds_enc.map(lambda x:{"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
          .rename_column("float_labels","labels"))

print(ds_enc['train'].features)
print(len(tokenizer))
ds_enc.save_to_disk(r'./dataset/')
