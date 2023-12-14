import torch
from torch.nn import functional as F
import tensorflow_hub as hub
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bertify_example(example, max_length=512):
    question = tokenizer.tokenize(example['question'])
    reference = tokenizer.tokenize(example['reference'])
    candidate = tokenizer.tokenize(example['candidate'])
    
    tokens = ['[CLS]'] + candidate + ['[SEP]'] + reference + ['[SEP]'] + question + ['[SEP]']
    
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
    segment_ids = torch.tensor([0] * (len(candidate) + 2) + [1] * (len(reference) + 1) + [2] * (len(question) + 1))

    input_ids = F.pad(torch.tensor(input_ids), (0, max_length - len(input_ids)), value=0)
    segment_ids = F.pad(torch.tensor(segment_ids), (0, max_length - len(segment_ids)), value=0)
    
    return {'input_ids': input_ids, 'segment_ids': segment_ids}


def bertify_examples(examples, max_length=512):
    input_ids = []
    segment_ids = []
    for example in examples:
        example_inputs = bertify_example(example, max_length=max_length)
        input_ids.append(example_inputs['input_ids'])
        segment_ids.append(example_inputs['segment_ids'])

    return {'input_ids': torch.stack(input_ids), 'segment_ids': torch.stack(segment_ids)}

# Loading the TensorFlow Hub model
bem = hub.load('https://tfhub.dev/google/answer_equivalence/bem/1')

# Example
examples = [{
    'question': 'why is the sky blue',
    'reference': 'light scattering',
    'candidate': 'scattering of light'
}]

# Prepare inputs in PyTorch format
inputs = bertify_examples(examples, max_length=tokenizer.model_max_length)

# The outputs are raw logits.
raw_outputs = bem(inputs)
raw_outputs_torch = torch.from_numpy(raw_outputs.numpy())
# They can be transformed into a classification 'probability' like so:
bem_score = float(F.softmax(raw_outputs_torch, dim=1)[:, 1].item())

print(f'BEM score: {bem_score}')
del bem
