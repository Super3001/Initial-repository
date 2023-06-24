import torch

check_ori = torch.load('model_gru1_2.ckpt')

from gru_1 import GRU_1, MyDataset

vocab_size = 30522
hidden_dim = 80  # # mbed_dim
num_layers = 3  # [1, 10]
dropout = 0.1  # [0.1, 0.2]
num_classes = 28
model = GRU_1(input_dim=vocab_size,
                           hidden_dim=hidden_dim,
                           num_layers=num_layers,
                           num_classes=num_classes,
                           dropout=dropout)

test_input = [
    'I am happy today.',
    'How are you',
    "I'm a little bit nervous",
    "what are you doing?",
    "today is a good day"
]

model_name = "microsoft/xtremedistil-l6-h384-uncased"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(s):
    return tokenizer(s, padding="max_length", truncation=True, max_length=64)

test_input = tokenize_function(test_input)

test_input = torch.tensor(test_input['input_ids'])

test_output = model(test_input)

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

for each in test_output:
    result = torch.softmax(each, dim=0)
    print(f'most likely:{emotions[int(torch.argmax(result))]} at {float(torch.max(result)):.2f}')