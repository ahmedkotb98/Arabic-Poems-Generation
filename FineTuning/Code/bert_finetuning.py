
import os
import random
import numpy as np
from tqdm import tqdm
import argparse
from configparser import ConfigParser
import bert_utils

import torch
from torch.utils.data import (DataLoader,Dataset, random_split, RandomSampler, SequentialSampler,
                              TensorDataset)

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForMaskedLM,BertTokenizer


def mask_tokens(inputs, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling:
    80% MASK, 10% random, 10% original.
    * The standard implementation from Huggingface Transformers library *
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    # MLM Prob is 0.15 in examples
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in
        labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(
        special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with
    # tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens
    # unchanged
    return inputs, labels

    
def accuracy(out, labels, total_test):
    class_preds = out.data.cpu().numpy().argmax(axis=-1)
    labels = labels.data.cpu().numpy()
    return np.sum(class_preds == labels) / total_test    


class BertFinetuning():
  def __init__(
      self,
      model,
      tokenizer,
      train_dataset,
      eval_dataset,
      batch_size,
      lr,
      adam_epsilon,
      epochs
  ):
      """
      :param model: Bert Model to train
      :param tokenizer: Bert Tokenizer to train
      :param train_dataset:
      :param eval_dataset:
      :param batch_size: Stick to 1 if not using using a high end GPU
      :param lr: Suggested learning rate from paper is 5e-5
      :param adam_epsilon: Used for weight decay fixed suggested parameter is
      1e-8
      :param epochs: Usually a single pass through the entire dataset is
      satisfactory
      :return: model_train , tokenizer_train ,Loss
      """
      self.model=model
      self.tokenizer=tokenizer
      self.train_dataset=train_dataset
      self.eval_dataset=eval_dataset
      self.batch_size=batch_size
      self.lr=lr
      self.adam_epsilon=adam_epsilon
      self.epochs=epochs


  def train_evaluate():
    train_sampler = RandomSampler(self.train_dataset)
    train_dataloader = DataLoader(
        self.train_dataset, sampler=train_sampler, batch_size=self.batch_size)

    t_total = len(train_dataloader) // self.batch_size  # Total Steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in self.model.named_parameters() if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in self.model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 0, t_total)

    # ToDo Case for fp16

    # Start of training loop
    print("***** Running training *****")
    print("  Num examples = ", len(self.train_dataset))
    print("  Batch size = ", self.batch_size)

    self.model.train()
    global_step = 0
    acc = 0
    total_test = 0
    tr_loss, logging_loss = 0.0, 0.0
    self.model.resize_token_embeddings(len(tokenizer))
    self.model.zero_grad()
    train_iterator = trange(int(self.epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            inputs, labels = mask_tokens(self.batch, self.tokenizer)
            inputs = inputs.to('cuda')  # Don't bother if you don't have a gpu
            labels = labels.to('cuda')

            loss, pred_masks = self.model(inputs, masked_lm_labels=labels)
            # model outputs are always tuple in transformers (see doc)
            loss = loss
            loss.backward()
            tr_loss += loss.item()
            total_test += labels.nelement()
            acc += accuracy(pred_masks, labels, total_test)

            # if (step + 1) % 1 == 0: # 1 here is a placeholder for gradient
            # accumulation steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

    print(" global_step = %s average loss = %s"%(global_step, tr_loss / global_step))
    print("Accuracy = " , 100 * acc)
    model_train=self.model
    tokenizer_train=self.tokenizer


    #########Evaluate#############
    self.model.eval()
    eval_sampler = SequentialSampler(self.eval_dataset)
    eval_dataloader = DataLoader(
        self.eval_dataset, sampler=eval_sampler, batch_size=self.batch_size)
    print("***** Running evaluation *****")
    print("  Num examples = %d", len(self.eval_dataset))
    print("  Batch size = %d", self.batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    acc = 0
    self.model.eval()
    total_test = 0
    # Evaluation loop
    for batch in tqdm_notebook(eval_dataloader, desc='Evaluating'):
      inputs, true_masks = mask_tokens(self.batch, self.tokenizer)
      inputs = inputs.to('cuda')
      true_masks = true_masks.to('cuda')
      with torch.no_grad():
        loss, pred_masks = self.model(inputs, masked_lm_labels=true_masks)
        lm_loss = loss
        eval_loss += lm_loss.mean().item()
        total_test += true_masks.nelement()
        acc += accuracy(pred_masks, true_masks, total_test)
      nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss)).item()
    result = {
        'perplexity': "{:.3f}".format(perplexity),
        'eval_loss': "{:.3f}".format(eval_loss),
        'acc': "{:.3f}".format(100 * acc)
    }
    return model_train , tokenizer_train , result




def main():
  parser = argparse.ArgumentParser(description="Finetuning Bert Model")
  parser.add_argument(
    "-i",
    "--train_dataset",
    help="Path your train dataset",
    required=True,
    type=str,
  )
  parser.add_argument(
    "-t",
    "--test_dataset",
    help="Path yout test dataset",
    required=True,
    type=str,
  )
  parser.add_argument(
    "-s",
    "--output_dir",
    help="Path to save the model files",
    required=True,
    type=str,
  )
  args = parser.parse_args()

  #read model name and hyperparamter from config file
  config_object = ConfigParser()
  config_object.read("config.ini")
  model_1 = config_object["arabert"]
  model_name = model_1["model_name"]
  epochs = model_1["epochs"]
  lr = model_1["lr"]
  adam_epsilon = model_1["adam_epsilon"]
  batch_size=model_1["batch_size"]
  
  #GPU Connection
  if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
  else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

  #load tokenizer and model
  tokenizer = bert_utils.load_tokenizer(model_name)
  model = bert_utils.load_model(model_name)
  model.cuda()

  #read train and test dataset
  dataset_train = bert_utils.create_dataset(tokenizer,args.train_dataset,block_size=24)
  dataset_test = bert_utils.create_dataset(tokenizer,args.test_dataset,block_size=24)

  #train_evaluate
  bert_utils.set_seed(seed=42)
  mlmfinetuning=BertFinetuning(
  model,
  tokenizer,
  dataset_train,
  dataset_test,
  batch_size,lr,
  adam_epsilon,
  epochs)
  model , tokenizer , eval_loss = mlmfinetuning.train_evaluate()
  print(eval_loss)
  
  #save model
  bert_utils.Save_model_after_training(model,tokenizer,args.output_dir)


if __name__ == "__main__":
    main()