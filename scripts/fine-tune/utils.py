import torch
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Used to turn .txt file into a suitable dataset object
    """

    def __init__(self, file_path, tokenizer, device, max_length=80 ):
        self.device = device
        with tf.io.gfile.GFile(file_path, "r") as f:
            text = f.read()
        lines = text.splitlines()
        self.samples = [
                        tokenizer.encode(line,
                                         max_length=max_length,
                                         add_special_tokens=True,
                                         pad_to_max_length=True)
                        for line in tqdm(lines)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return torch.tensor(self.samples[item]).to(self.device)


def mask_tokens(inputs, tokenizer, device):
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
        len(tokenizer), labels.shape, device=device, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens
    # unchanged
    return inputs, labels


def accuracy(out, labels, total):
    class_preds = out.data.cpu().numpy().argmax(axis=-1)
    labels = labels.data.cpu().numpy()
    return np.sum(class_preds == labels) / total


def dump_dataset(dataset, path):
    pickle.dump(
        dataset,
        open(path, 'wb')
    )


def load_dataset(path):
    return pickle.load(open(path, 'rb'))