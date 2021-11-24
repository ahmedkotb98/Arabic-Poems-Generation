# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""
import os
import collections
import random
import time
from multiprocessing import Pool

import tensorflow as tf
import six

import tokenization

flags = tf.flags
FLAGS = flags.FLAGS


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(
            self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next
    ):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (
            " ".join([str(x) for x in self.tokens])
        )
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions])
        )
        s += "masked_lm_labels: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_labels])
        )
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def print_example(instance, features):
    tf.logging.info("*** Example ***")
    tf.logging.info(
        "tokens: %s"
        % " ".join([str(x) for x in instance.tokens])
    )

    for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
            values = feature.int64_list.value
        elif feature.float_list.value:
            values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values]))
        )


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def transform(instance, tokenizer):
    """Transform instance to inputs for MLM and NSP."""
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= FLAGS.max_seq_length

    max_lens = FLAGS.max_seq_length
    input_ids.extend([0] * (max_lens - len(input_ids)))
    input_mask.extend([0] * (max_lens - len(input_mask)))
    segment_ids.extend([0] * (max_lens- len(segment_ids)))

    assert len(input_ids) == FLAGS.max_seq_length
    assert len(input_mask) == FLAGS.max_seq_length
    assert len(segment_ids) == FLAGS.max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    max_pred_seq = FLAGS.max_predictions_per_seq
    masked_lm_positions.extend([0] * (max_pred_seq - len(masked_lm_positions)))
    masked_lm_ids.extend([0] * (max_pred_seq - len(masked_lm_positions)))
    masked_lm_weights.extend([0.0] * (max_pred_seq - len(masked_lm_positions)))

    next_sentence_label = 1 if instance.is_random_next else 0
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    return features


def convert_to_tfexample(instances, tokenizer):
    """Create TF example files from `TrainingInstance`s."""
    tf_examples = []
    for inst_index, instance in enumerate(instances):
        features = transform(instance, tokenizer)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        tf_examples.append(tf_example)

        if inst_index < 2:
            print_example(instance, features)

    return tf_examples


def tokenize_lines(x):
    """Worker function to tokenize lines based on the tokenizer, and perform vocabulary lookup."""
    lines, tokenizer = x
    results = []
    for line in lines:
        if not line:
            break
        line = line.strip()
        # Empty lines are used as document delimiters
        if not line:
            results.append([])
        else:
            tokens = tokenizer.tokenize(line)
            if tokens:
                results.append(tokens)
    return results


def write_to_files(features, output_file):
    """Create TF example files from `TrainingInstance`s."""
    total_written = len(features)
    writer = tf.python_io.TFRecordWriter(output_file)
    for feature in features:
        writer.write(feature.SerializeToString())
    writer.close()

    tf.logging.info('Wrote %d total instances', total_written)


def create_training_instances(x):
    """Create `TrainingInstance`s from raw text."""
    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.

    (input_files, tokenizer, output_file) = x
    time_start = time.time()

    all_documents = [[]]
    for input_file in input_files:
        with tf.io.gfile.GFile(input_file, "r") as reader:
            lines = reader.read().split('\n')
            tokenized_results = tokenize_lines((lines, tokenizer))
            for tokenized_result in tokenized_results:
                for line in tokenized_result:
                    if not line:
                        all_documents.append([])
                    else:
                        all_documents[-1].append(line)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    # generate training instances
    vocab_words = list(tokenizer.vocab.keys())
    instances = []

    for _ in range(FLAGS.dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    (all_documents, document_index, vocab_words)))
    tfexample_instances = convert_to_tfexample(instances, tokenizer)

    # write output to files. Used when pre-generating files
    if output_file:
        tf.logging.info('*** Writing to output file %s ***', output_file)
        features = tfexample_instances
        write_to_files(features, output_file)
        features = None
    else:
        features = tfexample_instances

    time_end = time.time()
    tf.logging.info('Process %d files took %.1f s',
                    len(input_files), time_end - time_start)
    return features


def create_instances_from_document(x):
    """Creates `TrainingInstance`s for a single document."""
    (all_documents, document_index, vocab_words) = x
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = FLAGS.max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random.random() < FLAGS.short_seq_prob:
        target_seq_length = random.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or random.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = random.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                    tokens, vocab_words)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

def _is_start_piece_sp(piece):
  """Check if the current piece is the starting with (sentence piece)."""
  special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
  special_pieces.add(u"€".encode("utf-8"))
  special_pieces.add(u"£".encode("utf-8"))
  # Note(mingdachen):
  # For foreign characters, we always treat them as a whole piece.
  english_chars = set(list("abcdefghijklmnopqrstuvwxyz"))
  ARABIC_NORMALIZED_CHARACHTERS = set(list("0122334556778899ءءاااااااااااااااببببببببببببببتتتتتتتتتتتتتتتتتتتتثثثثثثثجججججججججججججججججججحححححخخخخخخخددددددددددددددذذذذذرررررررررررررزززسسسسسسششششششصصصصصضضضضضضططططظظظظظعععععغغغغغغفففففففففففففففففقققققققككككككككككككككككككككككككككككككككككللللللللممممننننننننننننننهههههههههههههههههههههوووووووووووووووووووووووووووويييييييييييييييييييييييييييييييييييييي"))

  if (six.ensure_str(piece).startswith("▁") or
      six.ensure_str(piece).startswith("<") or piece in special_pieces or
      not all([i.lower() in english_chars.union(special_pieces)
               for i in piece]) or
      not all([i in ARABIC_NORMALIZED_CHARACHTERS.union(special_pieces)
               for i in piece])
  ):
    return True

  return False


def create_masked_lm_predictions(tokens, vocab_words):
    """Creates the predictions for the masked LM objective."""
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
                not _is_start_piece_sp(token)):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    random.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(FLAGS.max_predictions_per_seq,
                         max(1, int(round(len(tokens) * FLAGS.masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    time_start = time.time()
    random.seed(FLAGS.random_seed)

    # create output dir
    output_dir = os.path.expanduser(FLAGS.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case,
        spm_model_file=FLAGS.spm_model_file
    )
    input_files = []
    for input_pattern in FLAGS.input_file.split(','):
        input_files.extend(tf.gfile.Glob(input_pattern))

    # Print files
    for input_file in input_files:
        tf.logging.info('\t%s', input_file)

    num_inputs = len(input_files)
    num_outputs = min(FLAGS.num_outputs, num_inputs)
    tf.logging.info('*** Reading from %d input files ***', num_inputs)

    # calculate the number of splits
    file_splits = []
    split_size = (num_inputs + num_outputs - 1) // num_outputs
    for i in range(num_outputs):
        split_start = i * split_size
        split_end = min(num_inputs, (i + 1) * split_size)
        file_splits.append(input_files[split_start:split_end])

    # prepare workload
    count = 0
    process_args = []
    for i, file_split in enumerate(file_splits):
        output_file = os.path.join(
            output_dir, 'part-{}.tfrecord'.format(str(i).zfill(3)))
        count += len(file_split)

        process_args.append((file_split, tokenizer, output_file))

    nworker = FLAGS.num_workers
    if nworker > 1:
        pool = Pool(processes=nworker)
        pool.map(create_training_instances, process_args)
        pool.close()
        pool.join()
    else:
        for process_arg in process_args:
            create_training_instances(process_arg)

    # sanity check
    assert count == len(input_files)

    time_end = time.time()
    tf.logging.info('Time cost=%.1f', time_end - time_start)


if __name__ == "__main__":
    flags.DEFINE_string(
        "input_file", None, 'Input files, separated by comma. For example, "~/data/*.txt"'
    )

    flags.DEFINE_string(
        "output_dir", None, "Output TF records directory"
    )

    flags.DEFINE_string("vocab_file", None, "The path of vocab file.")

    flags.DEFINE_string("spm_model_file", None, "The model file for sentence piece tokenization.")

    flags.DEFINE_bool(
        "do_lower_case",
        True,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.",
    )

    flags.DEFINE_bool(
        "do_whole_word_mask",
        False,
        "Whether to use whole word masking rather than per-WordPiece masking.",
    )

    flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

    flags.DEFINE_integer(
        "max_predictions_per_seq",
        20,
        "Maximum number of masked LM predictions per sequence.",
    )

    flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

    flags.DEFINE_integer(
        "num_workers",
        8,
        "Number of workers for parallel processing, where each generates an output file.")

    flags.DEFINE_integer(
        "num_outputs",
        1,
        "Number of workers for parallel processing, where each generates an output file.")

    flags.DEFINE_integer(
        "dupe_factor",
        10,
        "Number of times to duplicate the input data (with different masks).",
    )

    flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

    flags.DEFINE_float(
        "short_seq_prob",
        0.1,
        "Probability of creating sequences which are shorter than the " "maximum length.",
    )
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("spm_model_file")
    tf.app.run()


