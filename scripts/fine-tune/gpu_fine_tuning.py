import os
import subprocess
import logging
import random
import gc
import time
import argparse
from importlib import reload

import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from transformers import BertForMaskedLM, BertTokenizer, XLNetTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import TextDataset, mask_tokens, accuracy, dump_dataset, load_dataset


class GPUFineTuning:
    def __init__(self,
                train_dataset_path,
                test_dataset_path,
                output_dir,
                model_name,
                checkpoint_path,
                bert_config,
                tokenizer,
                learning_rate,
                epochs,
                batch_size,
                max_sequence_len,
                adam_epsilon,
                device,
    ):
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.bert_config = bert_config
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_sequence_len = max_sequence_len
        self.device = device
        self.adam_epsilon = adam_epsilon
        self.model_dir = os.path.join(self.output_dir, 'saved_model')

    def checkpoint(self):
        TF_OUTPUT_PATH = os.path.join(self.output_dir, "tf_checkpoints")
        TORCH_OUTPUT_PATH = os.path.join(self.output_dir, "torch_checkpoints")
        checkpoint_name = self.checkpoint_path.split('/')[-1]
        pt_path = os.path.join(TORCH_OUTPUT_PATH, 'torch_' + checkpoint_name + '.bin')

        if os.path.exists(pt_path):
            logging.info("Found existing Pytorch checkpoint {}".format(pt_path))
            return pt_path

        if not os.path.exists(TF_OUTPUT_PATH):
            os.makedirs(TF_OUTPUT_PATH)

        if not os.path.exists(TORCH_OUTPUT_PATH):
            os.makedirs(TORCH_OUTPUT_PATH)

        logging.info("Downloading Tensorflow checkpoints ...")
        subprocess.call(['gsutil', 'cp', self.checkpoint_path + '.*', TF_OUTPUT_PATH])

        logging.info("Converting Tensorflow checkpoints to Pytorch...")
        tf_path = os.path.join(TF_OUTPUT_PATH, checkpoint_name)
        subprocess.call(['python3', '-m', 'pytorch_pretrained_bert', 'convert_tf_checkpoint_to_pytorch',
                        tf_path, self.bert_config, pt_path])

        subprocess.call(['rm', '-rf', TF_OUTPUT_PATH])
        logging.info("Converted Successfully {}".format(pt_path))

        return pt_path

    def load_model_tokenizer(self):
        if self.model_name:
            logging.info('Loading BERT tokenizer...')
            tokenizer = BertTokenizer.from_pretrained(self.model_name,
                                                      do_lower_case=False)
            logging.info('Loading BERT pre-trained model from transformers Hub...')
            model = BertForMaskedLM.from_pretrained(
                self.model_name,
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )

        else:
            logging.info('Loading BERT tokenizer...')
            tokenizer = XLNetTokenizer.from_pretrained(self.tokenizer,
                                                       do_lower_case=False)

            logging.info('Loading BERT pre-trained model from checkpoint...')
            model_checkpoint = self.checkpoint()
            model = BertForMaskedLM.from_pretrained(
                model_checkpoint,
                config=self.bert_config,
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )

        return model, tokenizer

    def prepare_train_test_datasets(self, tokenizer):
        train_path = os.path.join(self.output_dir, 'train_dataset.pkl')
        if os.path.exists(train_path):
            logging.info('Loading training dataset from {}'.format(train_path))
            encoded_train_data = load_dataset(train_path)
        else:
            logging.info('Preparing training dataset...')
            encoded_train_data = TextDataset(tokenizer=tokenizer,
                                        file_path=self.train_dataset_path,
                                        max_length=self.max_sequence_len,
                                        device=self.device)
            dump_dataset(encoded_train_data, train_path)
            logging.info('Saved the tokenized training dataset to {}'.format(train_path))

        train_dataloader = DataLoader(encoded_train_data,
                                      sampler=RandomSampler(encoded_train_data),
                                      batch_size=self.batch_size)

        test_path = os.path.join(self.output_dir, 'test_dataset.pkl')
        if os.path.exists(test_path):
            logging.info('Loading test dataset from {}'.format(test_path))
            encoded_test_data = load_dataset(test_path)
        else:
            logging.info('Preparing test dataset...')
            encoded_test_data = TextDataset(tokenizer=tokenizer,
                                        file_path=self.test_dataset_path,
                                        max_length=self.max_sequence_len,
                                        device=self.device)
            dump_dataset(encoded_test_data, test_path)
            logging.info('Saved the tokenized test dataset to {}'.format(test_path))

        test_dataloader = DataLoader(encoded_test_data,
                                      sampler=SequentialSampler(encoded_test_data),
                                      batch_size=self.batch_size)

        return train_dataloader, test_dataloader

    def run(self):
        model, tokenizer = self.load_model_tokenizer()
        train_dataloader, test_dataloader = self.prepare_train_test_datasets(tokenizer)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.learning_rate,
                          eps=self.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 0, len(train_dataloader) // self.batch_size)

        logging.info('Training Starting ...')
        model = model.to(self.device)
        best_score = float('-inf')
        best_param_score = None
        best_epoch_score = None
        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss = self.train_model(train_dataloader,
                                     tokenizer, model, optimizer,
                                     scheduler)
            end_time = time.time()
            test_loss, acc = self.val_model(test_dataloader,
                                       model, tokenizer)

            logging.info(
                "Epoch {} Training loss: {:.3f} Epoch time: {} seconds".format(
                    epoch + 1, train_loss, round((end_time - start_time), 3)
                )
            )

            print(
                "Epoch {} Training loss: {:.3f} Epoch time: {} seconds".format(
                    epoch + 1, train_loss, round((end_time - start_time), 3)
                )
            )

            perplexity = torch.exp(torch.tensor(test_loss)).item()
            accuracy = 100 * acc
            print("Testing loss: {:.3f}".format(test_loss))
            print("Testing Perplexity : {:.3f}".format(perplexity))
            print("Testing Accuracy : {:.3f}".format(accuracy))

            if best_score < accuracy:
                best_score = accuracy
                best_param_score = model.state_dict()
                best_epoch_score = epoch

        print('Best model came from Epoch {} with score of {}'.format(best_epoch_score + 1,
                                                                             best_score))
        logging.info('Best model came from Epoch {} with score of {}'.format(best_epoch_score + 1,
                                                                             best_score))
        model.load_state_dict(best_param_score)
        logging.info("Saving model to : {}".format(self.output_dir))
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        del model
        torch.cuda.empty_cache()

    def train_model(self, train_dataloader, tokenizer, model, optimizer, scheduler):
        tr_loss = 0.0
        model.train()
        model.resize_token_embeddings(len(tokenizer))
        model.zero_grad()
        for batch in tqdm(train_dataloader, desc="Train Iteration"):
            inputs, true_masks = mask_tokens(batch, tokenizer, self.device)
            inputs = inputs.to(self.device)
            true_masks = true_masks.to(self.device)
            outputs = model(inputs, labels=true_masks)
            loss = outputs.loss

            loss.backward()
            tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            del inputs, true_masks

        gc.collect()
        loss = tr_loss / len(train_dataloader)
        return loss

    def val_model(self, test_loader, model, tokenizer):
        test_loss = 0.0
        total_test = 0
        acc = 0.0
        model.eval()
        for batch in tqdm(test_loader, desc="Test Iteration"):
            inputs, true_masks = mask_tokens(batch, tokenizer, self.device)
            inputs = inputs.to(self.device)
            true_masks = true_masks.to(self.device)
            with torch.no_grad():
                outputs = model(inputs, labels=true_masks)
                loss = outputs.loss
                logits = outputs.logits

                test_loss += loss.item()
                total_test += true_masks.nelement()
                acc += accuracy(logits, true_masks, total_test)

        loss = test_loss / len(test_loader)
        return loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--train_dataset_path',
      type=str,
      required=True,
      default=None,
      help='The training dataset file path',
    )

    parser.add_argument(
      '--test_dataset_path',
      type=str,
      required=True,
      default=None,
      help='The test dataset file path',
    )

    parser.add_argument(
      '--output_dir',
      type=str,
      required=True,
      default=None,
      help='fine-tuning output folder path',
    )

    parser.add_argument(
      '--model_name',
      type=str,
      required=False,
      default=None,
      help='Model name from transformers hub',
    )

    parser.add_argument(
      '--checkpoint_path',
      type=str,
      required=False,
      default=None,
      help='Google cloud storage checkpoint path ',
    )

    parser.add_argument(
      '--bert_config',
      type=str,
      required=False,
      default=None,
      help='The config json file corresponding to the pre-trained BERT model.',
    )

    parser.add_argument(
      '--tokenizer',
      type=str,
      required=False,
      default=None,
      help='The tokenizer model file path',
    )

    parser.add_argument(
      '--learning_rate',
      type=int,
      required=False,
      default=5e-5,
      help='Learning rate',
    )

    parser.add_argument(
      '--adam_epsilon',
      type=int,
      required=False,
      default=1e-8,
      help='Adam epsilon value',
    )

    parser.add_argument(
      '--epochs',
      type=int,
      required=False,
      default=4,
      help='Number of epochs',
    )

    parser.add_argument(
      '--batch_size',
      type=int,
      required=False,
      default=32,
      help='Batch size',
    )

    parser.add_argument(
      '--max_sequence_len',
      type=int,
      required=False,
      default=80,
      help='Max sequence length',
    )

    parser.add_argument(
      '--seed',
      type=int,
      required=False,
      default=32,
      help='Random seed value',
    )
    args = parser.parse_args()
    print("*" * 100)

    # create new folder for model outputs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    reload(logging)
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
        filename=os.path.join(args.output_dir, "logging.log")
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    gpu_fineTune = GPUFineTuning(
        train_dataset_path=args.train_dataset_path,
        test_dataset_path=args.test_dataset_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        bert_config=args.bert_config,
        tokenizer=args.tokenizer,
        batch_size=args.batch_size,
        max_sequence_len=args.max_sequence_len,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        device=device
    )

    gpu_fineTune.run()