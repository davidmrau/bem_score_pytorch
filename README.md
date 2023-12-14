# Answer Equivalence BEM Score

## Overview
In the following you will find and example of running the Answer Equivalence BEM score using the **Huggingface Tokenizer** and **Pytorch**.

## Example

```python
import torch
import tensorflow_hub as hub
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bertify_example(example, max_length=512):
    question = tokenizer.tokenize(example['question'])
    reference = tokenizer.tokenize(example['reference'])
    candidate = tokenizer.tokenize(example['candidate'])

    tokens = ['[CLS]'] + candidate + ['[SEP]'] + reference + ['[SEP]'] + question + ['[SEP]']
    
    inputs = tokenizer(tokens, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
    inputs = {key: value[0] for key, value in inputs.items()}  # Unpack from batch dimension

    return {'input_ids': inputs['input_ids'], 'segment_ids': inputs['token_type_ids']}

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
inputs = bertify_examples(examples)

# The outputs are raw logits.
raw_outputs = bem(inputs)
raw_outputs_torch = torch.from_numpy(raw_outputs.numpy())
print(raw_outputs_torch)
# They can be transformed into a classification 'probability' like so:
bem_score = torch.nn.functional.F.softmax(raw_outputs_torch, dim=0)
print(bem_score)

print(f'BEM score: {bem_score}')


```

BEM score: 0.9891803860664368

## Credits

Please see original paper [Tomayto, Tomahto. Beyond Token-level Answer Equivalence for Question Answering Evaluation](https://arxiv.org/abs/2202.07654) 

Original implementation in Tensorflow: 

https://www.kaggle.com/models/google/bert/frameworks/TensorFlow2/variations/answer-equivalence-bem/versions/1
