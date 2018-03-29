import os.path
import argparse


class AttrDict(dict):
    """
    Dictionary whose elements can be accessed as attributes
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self  # black magic


def get_base_config():
    """
    :return: Hyper-parameters of the model
    """
    config = AttrDict()

    # integer parameters
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620
    config['src_vocab_size'] = 30000
    config['trg_vocab_size'] = 30000

    config['batch_size'] = 80
    config['sort_k_batches'] = 12
    config['beam_size'] = 12

    config['finish_after'] = 1000000  # maximum number of updates
    config['max_epochs'] = 0          # TODO
    config['save_freq'] = 500         # save model after this many updates
    config['sampling_freq'] = 13      # show samples from model after this many updates
    config['hook_samples'] = 2        # show this many samples at each sampling
    config['bleu_val_freq'] = 5000    # validate bleu after this many updates
    config['val_burn_in'] = 80000     # start bleu validation after this many updates

    # boolean parameters
    config['normalized_bleu'] = 1     # normalize cost according to sequence length after beam-search
    config['reload'] = 1              # reload model from files if it exists
    config['load_weights'] = 0        # TODO
    config['load_embeddings'] = 0     # looks for pre-trained embeddings in data dir
    config['fix_embeddings'] = 0      # fix embedding weights during training
    config['train_val'] = 1           # BLEU evaluation on a subset of training corpus
    config['remove_eos'] = 1          # remove sentence terminators in the sampling output (more fair
    config['remove_unk'] = 0          # remove unk tokens in the sampling output

    # real-valued parameters
    config['step_clipping'] = 1.0
    config['weight_scale'] = 0.01
    config['weight_noise_ff'] = 0.0
    config['weight_noise_rec'] = 0.0
    config['dropout'] = 0.0            # dropout rate
    config['train_noise'] = 0.0        # input noise (equivalent to dropout on the first layer)
    config['dev_noise'] = 0.0

    return config


def finalize_config(config):
    data_dir = config['datadir']
    output_dir = config['saveto']

    config['val_bleu_out'] = 'val_bleu_scores.npz'
    config['bleu_script'] = os.path.join(data_dir, 'multi-bleu.perl')
    config['src_vocab'] = os.path.join(data_dir, 'vocab.src.pkl')
    config['trg_vocab'] = os.path.join(data_dir, 'vocab.trg.pkl')
    config['src_embed'] = os.path.join(data_dir, 'vectors.src.bin')
    config['trg_embed'] = os.path.join(data_dir, 'vectors.trg.bin')
    config['src_data'] = os.path.join(data_dir, 'train.src')
    config['trg_data'] = os.path.join(data_dir, 'train.trg')
    config['src_data_sample'] = os.path.join(data_dir, 'train.sample.src')
    config['trg_data_sample'] = os.path.join(data_dir, 'train.sample.trg')
    config['src_val'] = os.path.join(data_dir, 'dev.src')
    config['trg_val'] = os.path.join(data_dir, 'dev.trg')

    config['val_dev_out'] = os.path.join(output_dir, 'dev.out')
    config['val_best_dev_out'] = os.path.join(output_dir, 'dev.best.out')
    config['val_train_out'] = os.path.join(output_dir, 'train.sample.out')
    config['val_best_train_out'] = os.path.join(output_dir, 'train.sample.best.out')

    config['unk_id'] = 0
    config['eos_id'] = 1
    config['bos_id'] = 2

    config['bos_token'] = '<S>'  # TODO: fix hard-coding of those
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'


def parse_config():
    config = AttrDict(**get_base_config())  # to use config dict as a namespace
    parser = argparse.ArgumentParser()

    parser.add_argument('datadir', metavar='data-dir')
    parser.add_argument('saveto', metavar='output-dir')

    for k, v in config.items():
        assert type(v) in (int, float), 'Unknown type {} for parameter {}'.format(type(v), k)
        parser.add_argument('--{}'.format(k.replace('_', '-')), type=type(v), default=v)

    parser.parse_args(namespace=config)
    finalize_config(config)
    return config
