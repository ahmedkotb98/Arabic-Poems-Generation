import os
import random
import numpy as np
from tqdm import tqdm
import re
import string

import torch
from torch.utils.data import (DataLoader,Dataset, random_split, RandomSampler, SequentialSampler,
                              TensorDataset)

from transformers import BertForMaskedLM,BertTokenizer



class TextDataset(Dataset):
    """
    Used to turn .txt file into a suitable dataset object
    """

    def __init__(self, tokenizer, file_path, block_size=24):
        assert os.path.isfile(file_path)
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        lines = text.split("\n")
        self.examples = []
        for line in tqdm(lines):
            tokenized_text = tokenizer.encode(line, max_length=block_size,
                                              add_special_tokens=True, pad_to_max_length=True)  # Get ids from text
            self.examples.append(tokenized_text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])



def normalize_data(text):
  ARABIC_CHARS =   set(['ئ', 'ا', 'ب', 'ت', 'ة', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز',
                  'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م',
                  'ن', 'ه', 'و', 'ي', 'ء', 'آ', 'أ', 'ؤ', 'إ', 'ﺁ', 'ﺂ', 'ﺃ', 'ﺄ',
                  'ﺅ', 'ﺆ', 'ﺇ', 'ﺈ', 'ﺉ', 'ﺊ', 'ﺋ', 'ﺌ', 'ﺍ', 'ﺎ', 'ﺏ', 'ﺐ', 'ﺑ',
                  'ﺒ', 'ﺓ', 'ﺔ', 'ﺕ', 'ﺖ', 'ﺗ', 'ﺘ', 'ﺙ', 'ﺚ', 'ﺛ', 'ﺜ', 'ﺝ', 'ﺞ',
                  'ﺟ', 'ﺠ', 'ﺡ', 'ﺢ', 'ﺣ', 'ﺤ', 'ﺥ', 'ﺦ', 'ﺧ', 'ﺨ', 'ﺩ', 'ﺪ', 'ﺫ',
                  'ﺬ', 'ﺭ', 'ﺮ', 'ﺯ', 'ﺰ', 'ﺱ', 'ﺲ', 'ﺳ', 'ﺴ', 'ﺵ', 'ﺶ', 'ﺷ', 'ﺸ',
                  'ﺹ', 'ﺺ', 'ﺻ', 'ﺼ', 'ﺽ', 'ﺾ', 'ﺿ', 'ﻀ', 'ﻁ', 'ﻂ', 'ﻃ', 'ﻄ', 'ﻅ',
                  'ﻆ', 'ﻇ', 'ﻈ', 'ﻉ', 'ﻊ', 'ﻋ', 'ﻌ', 'ﻍ', 'ﻎ', 'ﻏ', 'ﻐ', 'ﻑ', 'ﻒ',
                  'ﻓ', 'ﻔ', 'ﻕ', 'ﻖ', 'ﻗ', 'ﻘ', 'ﻙ', 'ﻚ', 'ﻛ', 'ﻜ', 'ﻝ', 'ﻞ', 'ﻟ',
                  'ﻠ', 'ﻡ', 'ﻢ', 'ﻣ', 'ﻤ', 'ﻥ', 'ﻦ', 'ﻧ', 'ﻨ', 'ﻩ', 'ﻪ', 'ﻫ', 'ﻬ',
                  'ﻭ', 'ﻮ', 'ﻯ', 'ﻰ', 'ى', 'ﯼ', 'ﻱ', 'ﻲ', 'ﻳ', 'ﻴ', 'ﻵ', 'ﻶ', 'ﻷ',
                  'ﻸ', 'ﻹ', 'ﻺ', 'ﻻ', 'ﻼ'])
  
  CHARS_PREVENT_REMOVING = set(ARABIC_CHARS)
  STR_CHARS_PREVENT_REMOVING = ' '.join((c for c in CHARS_PREVENT_REMOVING))
  ESCAPED_CHARS_PREVENT_REMOVING = re.escape(STR_CHARS_PREVENT_REMOVING)
  return re.sub("[^%s]" % (ESCAPED_CHARS_PREVENT_REMOVING), '', text).strip()


def load_tokenizer(model_name):
  tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
  return tokenizer


def load_model(model_name):

  model = BertForMaskedLM.from_pretrained(
    model_name, 
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
  return model


def create_dataset(tokenizer, file_path, block_size=24):
    """
    Creates a dataset object from file path.
    :param tokenizer: Bert tokenizer to create dataset
    :param file_path: Path where data is stored
    :param block_size: Should be in range of [0,512], viable choices are 64,
    128, 256, 512
    :return: The dataset
    """
    dataset = TextDataset(tokenizer, file_path=file_path,
                          block_size=block_size)
    return dataset


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


def set_seed(seed=42):
    """
    Sets seed for all random number generators available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except:
        print('Cuda manual seed is not set')


def accuracy(out, labels, total_test):
    class_preds = out.data.cpu().numpy().argmax(axis=-1)
    labels = labels.data.cpu().numpy()
    return np.sum(class_preds == labels) / total_test


def train_evaluate(model , tokenizer , train_dataset , eval_dataset , batch_size, lr , adam_epsilon ,
          epochs):
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

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size)

    t_total = len(train_dataloader) // batch_size  # Total Steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 0, t_total)

    # ToDo Case for fp16

    # Start of training loop
    print("***** Running training *****")
    print("  Num examples = ", len(train_dataset))
    print("  Batch size = ", batch_size)

    model.train()
    global_step = 0
    acc = 0
    total_test = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    train_iterator = trange(int(epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            inputs, labels = mask_tokens(batch, tokenizer)
            inputs = inputs.to('cuda')  # Don't bother if you don't have a gpu
            labels = labels.to('cuda')

            loss, pred_masks = model(inputs, masked_lm_labels=labels)
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
    model_train=model
    tokenizer_train=tokenizer


    #########Evaluate#############
    model.eval()
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=batch_size)
    print("***** Running evaluation *****")
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    acc = 0
    model.eval()
    total_test = 0
    # Evaluation loop
    for batch in tqdm_notebook(eval_dataloader, desc='Evaluating'):
      inputs, true_masks = mask_tokens(batch, tokenizer)
      inputs = inputs.to('cuda')
      true_masks = true_masks.to('cuda')
      with torch.no_grad():
        loss, pred_masks = model(inputs, masked_lm_labels=true_masks)
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


def save_model_after_training(model,tokenizer,output_dir):
  # Create output directory if needed
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  print("Saving model to %s" % output_dir)
  # Save a trained model, configuration and tokenizer using `save_pretrained()`.
  # They can then be reloaded using `from_pretrained()`
  model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
  model_to_save.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)


def load_model_after_training(output_dir):

  model = BertForMaskedLM.from_pretrained(
    output_dir, 
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
  
  tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=False)
  return model , tokenizer