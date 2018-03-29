import numpy
import random

from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

from six.moves import cPickle


def get_vocab(vocab, vocab_size, unk_id=0, eos_id=1, bos_id=2):
    vocab = vocab if isinstance(vocab, dict) else cPickle.load(open(vocab))

    tokens_to_remove = [k for k, v in vocab.items() if v in [unk_id, eos_id, bos_id]]

    for token in tokens_to_remove:
        vocab.pop(token)
    vocab['<S>'] = bos_id
    vocab['</S>'] = eos_id
    vocab['<UNK>'] = unk_id

    return vocab


class PaddingWithEOS(Padding):
    """Padds a stream with given end of sequence idx."""
    def __init__(self, data_stream, eos_ids, **kwargs):
        kwargs['data_stream'] = data_stream
        self.eos_ids = eos_ids
        super(PaddingWithEOS, self).__init__(**kwargs)

    def get_data_from_batch(self, request=None):
        if request is not None:
            raise ValueError

        data = list(next(self.child_epoch_iterator))
        data_with_masks = []

        for i, (source, source_data) in enumerate(zip(self.data_stream.sources, data)):
            if source not in self.mask_sources:
                data_with_masks.append(source_data)
                continue

            shapes = [numpy.asarray(sample).shape for sample in source_data]
            lengths = [shape[0] for shape in shapes]
            max_sequence_length = max(lengths)
            rest_shape = shapes[0][1:]

            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")

            dtype = numpy.asarray(source_data[0]).dtype
            padded_data = numpy.ones((len(source_data), max_sequence_length) + rest_shape, dtype=dtype) * self.eos_ids[i]

            for i, sample in enumerate(source_data):
                padded_data[i, :len(sample)] = sample

            data_with_masks.append(padded_data)
            mask = numpy.zeros((len(source_data), max_sequence_length), self.mask_dtype)

            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1

            data_with_masks.append(mask)

        return tuple(data_with_masks)


def _length(sentences):
    return len(sentences[-1])


class _add_noise(object):
    def __init__(self, noise):
        self.noise = noise  # in [0,1), ratio of words to delete

    def __call__(self, sentence):
        if self.noise == 0:
            return sentence

        sentence, = sentence  # for whatever reason, a stream always yields a tuple

        n = int(len(sentence) * (1 - self.noise))  # number of words to keep
        n = max(n, 1)  # keep at least 1 word

        indices = random.sample(range(len(sentence)), n)

        return [x for i, x in enumerate(sentence) if i in indices],  # return tuple


class _not_too_long(object):
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, sentences):
        return all(len(sentence) <= self.seq_len for sentence in sentences)


def get_stream(vocab, data, vocab_size, unk_id, eos_id, bos_id, noise=0):
    vocab = get_vocab(vocab, vocab_size, unk_id, eos_id, bos_id)

    # Maps words to their index in the vocabulary. OOV words are replaced by <UNK> index.
    # Also appends </S> index at the end. No <S> token (TODO: bos_id parameter useless).
    dataset = TextFile([data], vocab, None)

    stream = Mapping(dataset.get_example_stream(), _add_noise(noise))
    stream.dataset = dataset  # for backward-compatibility
    return stream


def get_tr_stream(src_vocab, trg_vocab, src_data, trg_data,
                  src_vocab_size=30000, trg_vocab_size=30000,
                  unk_id=0, eos_id=1, bos_id=2, train_noise=0,
                  seq_len=50, batch_size=80, sort_k_batches=12, **kwargs):
    src_stream = get_stream(src_vocab, src_data, src_vocab_size, unk_id, eos_id, bos_id, train_noise)
    trg_stream = get_stream(trg_vocab, trg_data, trg_vocab_size, unk_id, eos_id, bos_id, 0)

    # Merge them to get a source, target pair
    stream = Merge([src_stream, trg_stream], ('source', 'target'))

    # Filter sequences that are too long
    stream = Filter(stream, predicate=_not_too_long(seq_len))

    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size * sort_k_batches))

    # Sort all samples in the read-ahead batch
    stream = Mapping(stream, SortMapping(_length))

    # Convert it into a stream again
    stream = Unpack(stream)

    # Construct batches from the stream with specified batch size
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    return PaddingWithEOS(stream, [eos_id, eos_id])


def get_dev_stream(val_set, src_vocab=None, src_vocab_size=30000,
                   unk_id=0, eos_id=1, bos_id=2, dev_noise=0, **kwargs):
    return get_stream(src_vocab, val_set, src_vocab_size, unk_id, eos_id, bos_id, dev_noise)


def get_stdin_stream(src_vocab=None, src_vocab_size=30000,
                     unk_id=0, eos_id=1, bos_id=2, **kwargs):
    # on Unix systems, '/proc/self/fd/0' is the file descriptor for standard input
    return get_stream(src_vocab, '/proc/self/fd/0', src_vocab_size, unk_id, eos_id, bos_id, 0)
