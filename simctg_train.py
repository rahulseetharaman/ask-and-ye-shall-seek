import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np
import logging
logging.getLogger().setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings('ignore')
from get_data import ClariqDataset
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)
from transformers import AdamW, get_linear_schedule_with_warmup
import time
from tqdm import tqdm
import datetime

import sys
sys.path.append("/work/pi_hzamani_umass_edu/rseetharaman/packages")
from simctg.lossfunction import SimCTGLoss
from simctg.simctgbart import SimCTGBART

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def train():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    epochs = 3
    learning_rate = 5e-5
    warmup_steps = 10
    epsilon = 1e-8

    # this produces sample output every 100 steps
    sample_every = 10

    model = SimCTGBART("prakharz/DIAL-BART0")
    tokenizer = model.tokenizer
    model = model.to(device)
    train_dataset = ClariqDataset(tokenizer=tokenizer, data_path="./train_bart.txt")
    val_dataset = ClariqDataset(tokenizer=tokenizer, data_path="./val_bart.txt")

    margin = 0.5
    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.bos_token_id
    simctgloss = SimCTGLoss(margin=margin, vocab_size=vocab_size, pad_token_id=pad_token_id)

    batch_size=4

    train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )

    
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    # This changes the learning rate as the training loop progresses
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = warmup_steps, 
                                                num_training_steps = total_steps)

    

    total_t0 = time.time()

    training_stats = []

    model = model.to(device)
    accum_iterations = 16

    for epoch_i in range(0, epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(tqdm(train_dataloader)):

            model.zero_grad()        

            input_ids = batch[0].to(device)
            attn_mask = batch[1].to(device)
            decoder_input_ids = batch[2].to(device)
            labels = batch[3].to(device)
            labels[labels[:, :] == tokenizer.pad_token_id] = -100

            # forward computation
            last_hidden_states, logits = model(encoder_inputs=input_ids, encoder_mask=attn_mask, decoder_inputs=decoder_input_ids, decoder_labels=labels)
            # loss computation

            mle_loss, cl_loss = simctgloss(last_hidden_states=last_hidden_states, logits=logits, 
                                        input_ids=decoder_input_ids, labels=labels)
            loss = mle_loss + cl_loss  

            batch_loss = loss.item()
            total_train_loss += batch_loss


            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

                model.eval()

                sample_outputs = model.model.generate(input_ids, decoder_start_token_id=tokenizer.pad_token_id, do_sample=True,   
                                    top_k=50, 
                                    max_length = 512,
                                    top_p=0.8, 
                                    penalty_alpha = 0.6,
                                    num_return_sequences=3)

                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
                
                model.train()

            loss.backward()

            if step % accum_iterations == 0:
                optimizer.step()
                scheduler.step()


        torch.save(model, "ctg_model.pt")

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)       
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
            
        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            input_ids = batch[0].to(device)
            attn_mask = batch[1].to(device)
            decoder_input_ids = batch[2].to(device)
            labels = batch[3].to(device)
            labels[labels[:, :] == tokenizer.pad_token_id] = -100

            with torch.no_grad():        

                # forward computation
                last_hidden_states, logits = model(encoder_inputs=input_ids, encoder_mask=attn_mask, decoder_inputs=decoder_input_ids, decoder_labels=labels)
                # loss computation

                mle_loss, cl_loss = simctgloss(last_hidden_states=last_hidden_states, logits=logits, 
                                            input_ids=decoder_input_ids, labels=labels)
                loss = mle_loss + cl_loss  
  
            batch_loss = loss.item()
            total_eval_loss += batch_loss        

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        validation_time = format_time(time.time() - t0)    

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


if __name__ == "__main__":
    train()

