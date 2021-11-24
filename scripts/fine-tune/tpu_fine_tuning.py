import os
import subprocess
import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename="fine_tuning_tpu_1.log",
)

import random
import gc
import time
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer, XLNetTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp # To read more, http://pytorch.org/xla/index.html#running-on-multiple-xla-devices-with-multiprocessing
import torch_xla.distributed.parallel_loader as pl
from utils import TextDataset, mask_tokens, accuracy, dump_dataset, load_dataset


class TPUFineTuning:
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
        self.learning_rate = learning_rate * xm.xrt_world_size() # Scale learning rate to num cores
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_sequence_len = max_sequence_len
        self.device = device
        self.adam_epsilon = adam_epsilon
        self.model_dir = os.path.join(self.output_dir, 'saved_model')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def checkpoint(self):
        TF_OUTPUT_PATH = os.path.join(self.output_dir, "tf_checkpoints")
        TORCH_OUTPUT_PATH = os.path.join(self.output_dir, "torch_checkpoints")
        if not os.path.exists(TF_OUTPUT_PATH):
            os.makedirs(TF_OUTPUT_PATH)

        if not os.path.exists(TORCH_OUTPUT_PATH):
            os.makedirs(TORCH_OUTPUT_PATH)

        checkpoint_name = self.checkpoint_path.split('/')[-1]
        logging.info("Downloading Tensorflow checkpoints ...")
        subprocess.call(['gsutil', 'cp', self.checkpoint_path + '.*', TF_OUTPUT_PATH])

        logging.info("Converting Tensorflow checkpoints to Pytorch...")
        tf_path = os.path.join(TF_OUTPUT_PATH, checkpoint_name)
        pt_path = os.path.join(TORCH_OUTPUT_PATH, 'torch_' + checkpoint_name + '.bin')
        subprocess.call(['python3', '-m', 'pytorch_pretrained_bert', 'convert_tf_checkpoint_to_pytorch',
                        tf_path, self.bert_config, pt_path])

        subprocess.call(['rm', '-rf', TF_OUTPUT_PATH])
        logging.info("Converted Successfully {}".format(pt_path))

        return pt_path

    def load_model_tokenizer(self):
        if self.model_name:
            logging.info('Loading BERT tokenizer...')
            tokenizer = BertTokenizer.from_pretrained(self.tokenizer,
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
            dataset_train = load_dataset(train_path)
        else:
            logging.info('Preparing training dataset...')
            dataset_train = TextDataset(tokenizer=tokenizer,
                                        file_path=self.train_dataset_path,
                                        device=device,
                                        max_length=self.max_sequence_len)
            dump_dataset(dataset_train, train_path)
            logging.info('Saved the tokenized training dataset to {}'.format(train_path))

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True,
        )

        train_dataloader = DataLoader(
            dataset_train,
            sampler=train_sampler,
            batch_size=self.batch_size,
            drop_last=True
        )

        test_path = os.path.join(self.output_dir, 'test_dataset.pkl')
        if os.path.exists(test_path):
            logging.info('Loading test dataset from {}'.format(test_path))
            dataset_test = load_dataset(test_path)
        else:
            logging.info('Preparing test dataset...')
            dataset_test = TextDataset(tokenizer=tokenizer,
                                       file_path=self.test_dataset_path,
                                       device=device,
                                       max_length=self.max_sequence_len)
            dump_dataset(dataset_test, test_path)
            logging.info('Saved the tokenized test dataset to {}'.format(test_path))

        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True,
        )

        test_dataloader = DataLoader(
            dataset_test,
            sampler=test_sampler,
            batch_size=self.batch_size,
            drop_last=True
        )

        return train_dataloader, test_dataloader

    def run(self, index=None):
        def train_model(train_loader, tokenizer, model, optimizer, scheduler=None):
            tr_loss = 0.0
            tracker = xm.RateTracker()
            model.train()
            for idx, batch in enumerate(train_loader):
                inputs, true_masks = mask_tokens(batch, tokenizer, self.device)
                inputs = inputs.to(device)
                true_masks = true_masks.to(device)
                outputs = model(inputs, labels=true_masks)
                loss = outputs.loss

                loss.backward()
                tr_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                xm.optimizer_step(optimizer)
                if scheduler is not None:
                    scheduler.step()
                model.zero_grad()
                if idx % 100 == 0:
                    logging.info('[xla:{}] ({}) Loss={:.3f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
                        xm.get_ordinal(), idx, loss.item(), tracker.rate(),
                        tracker.global_rate(), time.asctime()), flush=True
                    )
                del inputs, true_masks

            loss = tr_loss / len(train_loader)
            logging.info("global_step: {} loss: {:.3f}".format(len(train_loader), loss))
            gc.collect()

        def val_model(val_loader, model, tokenizer):
            test_loss = 0.0
            total_test = 0
            acc = 0
            model.eval()
            for idx, batch in enumerate(val_loader):
                inputs, true_masks = mask_tokens(batch, tokenizer, self.device)
                inputs = inputs.to(device)
                true_masks = true_masks.to(device)
                outputs = model(inputs, labels=true_masks)
                loss = outputs.loss
                logits = outputs.logits

                test_loss += loss.mean().item()
                total_test += true_masks.nelement()
                acc += accuracy(logits, true_masks, total_test)

            eval_loss = test_loss / len(val_loader)
            perplexity = torch.exp(torch.tensor(eval_loss)).item()
            logging.info("Eval Perplexity : {}".format("{:.3f}".format(perplexity)))
            logging.info("Eval loss : {}".format("{:.3f}".format(eval_loss)))
            logging.info("Eval Accuracy : {}".format("{:.3f}".format(100 * acc)))

            return eval_loss

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
            xm.master_print("Training Epoch.... {}".format(epoch))

            train_para_loader = pl.ParallelLoader(train_dataloader, [device])
            train_model(train_para_loader.per_device_loader(device),
                        tokenizer, model, optimizer,
                        scheduler=scheduler
                        )

            valid_para_loader = pl.ParallelLoader(test_dataloader, [device])
            eval_loss = val_model(valid_para_loader.per_device_loader(device),
                                  model, tokenizer
                                  )
            if best_score < eval_loss:
                best_score = eval_loss
                best_param_score = model.state_dict()
                best_epoch_score = epoch

        xm.master_print('Best model came from Epoch {} with score of {}'.format(best_epoch_score + 1,
                                                                                best_score))
        model.load_state_dict(best_param_score)

        logging.info("Saving model to : {}".format(self.output_dir))
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)


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
      required=True,
      default=None,
      help='Google cloud storage checkpoint path ',
    )

    parser.add_argument(
      '--bert_config',
      type=str,
      required=True,
      default=None,
      help='The config json file corresponding to the pre-trained BERT model.',
    )

    parser.add_argument(
      '--tokenizer',
      type=str,
      required=True,
      default=None,
      help='The tokenizer model file path',
    )

    parser.add_argument(
      '--tpu_ip',
      type=str,
      required=True,
      default=None,
      help='tpu internal IP',
    )

    parser.add_argument(
      '--num_tpu_cores',
      type=int,
      required=False,
      default=8,
      help='Number of tpu cores (e.g 8 cores with tpu v3.8)',
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

    os.system('export XRT_TPU_CONFIG="tpu_worker;0;{}:8470"'.format(args.tpu_ip))
    device = xm.xla_device()
    logging.info("TPU ID ADDRESS {}".format(args.tpu_ip))

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    tpu_fineTune = TPUFineTuning(
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

    xmp.spawn(tpu_fineTune.run(), nprocs=args.num_tpu_cores,
              start_method='fork')