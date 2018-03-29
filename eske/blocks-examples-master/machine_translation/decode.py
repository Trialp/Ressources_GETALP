import logging
import pprint
from machine_translation.configurations import parse_config

import os.path
import numpy as np
from theano import tensor
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.model import Model
from blocks.search import BeamSearch
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter

from machine_translation.model import BidirectionalEncoder, Decoder
from machine_translation.stream import get_stdin_stream, get_vocab
from machine_translation.checkpoint import SaveLoadUtils


logging.getLogger('').handlers = []  # reset logger
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

config = parse_config()
logger.info('Configuration:\n{}'.format(pprint.pformat(config)))

# Create Theano variables
logger.info('Creating theano variables')
source_sentence = tensor.lmatrix('source')
source_sentence_mask = tensor.matrix('source_mask')
target_sentence = tensor.lmatrix('target')
target_sentence_mask = tensor.matrix('target_mask')
sampling_input = tensor.lmatrix('input')

# Construct model
logger.info('Building RNN encoder-decoder')
encoder = BidirectionalEncoder(
    config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])
decoder = Decoder(
    config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
    config['enc_nhids'] * 2)

logger.info('Creating computational graph')

# Initialize model
logger.info('Initializing model')
encoder.weights_init = decoder.weights_init = IsotropicGaussian(config['weight_scale'])
encoder.biases_init = decoder.biases_init = Constant(0)
encoder.push_initialization_config()
decoder.push_initialization_config()
encoder.bidir.prototype.weights_init = Orthogonal()
decoder.transition.weights_init = Orthogonal()
encoder.initialize()
decoder.initialize()

logger.info("Building sampling model")
sampling_representation = encoder.apply(sampling_input, tensor.ones(sampling_input.shape))
generated = decoder.generate(sampling_input, sampling_representation)
search_model = Model(generated)

params = search_model.get_parameter_dict()
param_values = SaveLoadUtils().load_parameter_values(os.path.join(config['saveto'], 'params.npz'))
for k in params:
    params[k].set_value(param_values[k])

_, samples = VariableFilter(bricks=[decoder.sequence_generator], name="outputs")(ComputationGraph(generated[1]))
beam_search = BeamSearch(samples=samples)

# Read from standard input
stream = get_stdin_stream(**config)

vocab = get_vocab(config['trg_vocab'], config['trg_vocab_size'], config['unk_id'], config['eos_id'], config['bos_id'])
inv_vocab = {v: k for k, v in vocab.iteritems()}

unk_id = config['unk_id']
eos_id = config['eos_id']

for sample in stream.get_epoch_iterator():
    seq = sample[0]
    input_ = np.tile(seq, (config['beam_size'], 1))

    trans, costs = beam_search.search(
            input_values={sampling_input: input_},
            max_length=3 * len(seq), eol_symbol=eos_id,
            ignore_first_eol=True)

    trans_indices = [idx for idx in trans[0] if idx != eos_id]  # remove </S> from output
    trans_out = ' '.join(inv_vocab.get(idx, config['unk_token']) for idx in trans_indices)

    print trans_out
