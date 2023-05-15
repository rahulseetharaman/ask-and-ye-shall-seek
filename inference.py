import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from get_data import ABGCoQADataset
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
import numpy as np

torch.manual_seed(42)

def generate_one(tokenizer, model, prompt):
    generated = tokenizer(prompt, return_tensors='pt')
    input_ids = generated['input_ids'].to(device)
    sample_outputs = model.generate(
                                    input_ids,
                                    decoder_start_token_id=tokenizer.pad_token_id,
                                    do_sample=True,  
                                    temperature=0.9, 
                                    no_repeat_ngram_size=2, 
                                    top_k=500, 
                                    max_length = 512,
                                    top_p=0.95, 
                                    num_return_sequences=5
                                    )
    context = prompt.replace('.', '.\n')
    print(f"CONTEXT: {context}")
    print('-'*100)
    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))


def infer(tokenizer, model, split):
    batch_size = 4
    test_dataset = ABGCoQADataset(tokenizer=tokenizer, data_path=f"./{split}_t5.txt")
    test_dataloader = DataLoader(
                test_dataset, # The validation samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    all = []

    for step, batch in enumerate(tqdm(test_dataloader)):

        input_ids = batch[0].to(device)

        outputs = model.generate(input_ids, decoder_start_token_id=tokenizer.pad_token_id,
                                    do_sample=True,  
                                    temperature=0.9, 
                                    top_k=0, 
                                    max_length = 512,
                                    num_return_sequences=2)
        
        sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all.extend(sentences)
    
    return all
        

if __name__ == '__main__':
    device=torch.device('cuda')
    tokenizer = T5Tokenizer.from_pretrained("t5_tokenizer")
    model = T5ForConditionalGeneration.from_pretrained("t5_model").to(device)
    splits = ['val']
    for s in splits:
        all_cq = infer(tokenizer, model, s)
        f=open(f'{s}_answers.txt', 'w', encoding='utf-8')
        for a in all_cq:
            f.write(a)
            f.write('\n')
