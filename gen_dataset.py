from datasets import load_dataset
import os
ds = load_dataset("manu/french_poetry")

output_file = 'french_poetry_dataset.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    for poem in ds['train']:
        text_lines = poem['text'].split('\n')[5:]
        f.write('\n'.join(text_lines))
        f.write("\n\n")