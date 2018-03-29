"""
Encoder-Decoder with search for machine translation.

In this demo, encoder-decoder architecture with attention mechanism is used for
machine translation. The attention mechanism is implemented according to
[BCB]_. The training data used is WMT15 Czech to English corpus, which you have
to download, preprocess and put to your 'datadir' in the config file. Note
that, you can use `prepare_data.py` script to download and apply all the
preprocessing steps needed automatically.  Please see `prepare_data.py` for
further options of preprocessing.

.. [BCB] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural
   Machine Translation by Jointly Learning to Align and Translate.
"""

import logging
import pprint
from machine_translation.configurations import parse_config

from collections import Counter
from theano import tensor
from toolz import merge

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta,
                               CompositeRule)
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector

from machine_translation.checkpoint import CheckpointNMT, LoadNMT, LoadParameters
from machine_translation.model import BidirectionalEncoder, Decoder
from machine_translation.sampling import Sampler, BleuValidator, BleuEvaluator
from machine_translation.stream import get_tr_stream, get_dev_stream
from machine_translation.embeddings import load_embeddings

logging.getLogger('').handlers = []  # reset logger
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

config = parse_config()
logger.info('Configuration:\n{}'.format(pprint.pformat(config)))

tr_stream = get_tr_stream(**config)
next(tr_stream.get_epoch_iterator())

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
cost = decoder.cost(
    encoder.apply(source_sentence, source_sentence_mask),
    source_sentence_mask, target_sentence, target_sentence_mask)

logger.info('Creating computational graph')
cg = ComputationGraph(cost)

# Initialize model
logger.info('Initializing model')
encoder.weights_init = decoder.weights_init = IsotropicGaussian(
    config['weight_scale'])
encoder.biases_init = decoder.biases_init = Constant(0)
encoder.push_initialization_config()
decoder.push_initialization_config()
encoder.bidir.prototype.weights_init = Orthogonal()
decoder.transition.weights_init = Orthogonal()
encoder.initialize()
decoder.initialize()

if config['load_embeddings']:
    logger.info('Loading embeddings')
    if 'src_embed' in config:
        embeddings = load_embeddings(config['src_embed'], config['enc_embed'], config['src_vocab'],
                                     config['src_vocab_size'], unk_id=config['unk_id'])
        encoder.lookup.W.set_value(encoder.lookup.W.get_value() + embeddings)
    if 'trg_embed' in config:
        embeddings = load_embeddings(config['trg_embed'], config['enc_embed'], config['trg_vocab'],
                                     config['trg_vocab_size'], unk_id=config['unk_id'])
        W = decoder.sequence_generator.readout.feedback_brick.lookup.W
        W.set_value(W.get_value() + embeddings)

# apply dropout for regularization
if config['dropout'] > 0.0:
    # dropout is applied to the output of maxout in ghog
    logger.info('Applying dropout')
    dropout_inputs = [x for x in cg.intermediary_variables if x.name == 'maxout_apply_output']
    cg = apply_dropout(cg, dropout_inputs, config['dropout'])

# Apply weight noise for regularization
if config['weight_noise_ff'] > 0.0:
    logger.info('Applying weight noise to ff layers')
    enc_params = Selector(encoder.lookup).get_params().values()
    enc_params += Selector(encoder.fwd_fork).get_params().values()
    enc_params += Selector(encoder.back_fork).get_params().values()
    dec_params = Selector(
        decoder.sequence_generator.readout).get_params().values()
    dec_params += Selector(
        decoder.sequence_generator.fork).get_params().values()
    dec_params += Selector(decoder.state_init).get_params().values()
    cg = apply_noise(cg, enc_params+dec_params, config['weight_noise_ff'])

# Print shapes
shapes = [param.get_value().shape for param in cg.parameters]
logger.info("Parameter shapes: ")
for shape, count in Counter(shapes).most_common():
    logger.info('    {:15}: {}'.format(shape, count))
logger.info("Total number of parameters: {}".format(len(shapes)))

# Print parameter names
enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                           Selector(decoder).get_parameters())
logger.info("Parameter names: ")
for name, value in enc_dec_param_dict.items():
    logger.info('    {:15}: {}'.format(value.get_value().shape, name))
logger.info("Total number of parameters: {}"
            .format(len(enc_dec_param_dict)))

# Print number of params
logger.info("Number of parameters: {}"
            .format(sum(reduce(lambda a, b: a * b, p.get_value().shape) for p in cg.parameters)))

# Set up training model
logger.info("Building model")
training_model = Model(cost)

# Set extensions
logger.info("Initializing extensions")
extensions = [
    FinishAfter(after_n_batches=config['finish_after']),
    TrainingDataMonitoring([cost], after_batch=True),
    Printing(after_batch=True),
    CheckpointNMT(config['saveto'],
                  every_n_batches=config['save_freq'])
]

# Set up beam search and sampling computation graphs if necessary
if config['hook_samples'] >= 1 or config['bleu_script'] is not None:
    logger.info("Building sampling model")
    sampling_representation = encoder.apply(sampling_input, tensor.ones(sampling_input.shape))
    generated = decoder.generate(sampling_input, sampling_representation)
    search_model = Model(generated)

    _, samples = VariableFilter(
        bricks=[decoder.sequence_generator], name="outputs")(
            ComputationGraph(generated[1]))  # generated[1] is next_outputs

# Add sampling
if config['hook_samples'] >= 1:
    logger.info("Building sampler")
    extensions.append(
        Sampler(model=search_model, data_stream=tr_stream,
                hook_samples=config['hook_samples'],
                every_n_batches=config['sampling_freq'],
                src_vocab_size=config['src_vocab_size']))

# Add early stopping based on bleu
if config['bleu_script'] is not None:
    logger.info("Building bleu validator")
    dev_stream = get_dev_stream(val_set=config['src_val'], **config)
    extensions.append(
        BleuValidator(sampling_input, samples=samples, config=config,
                      model=search_model, data_stream=dev_stream,
                      ground_truth=config['trg_val'],
                      normalize=config['normalized_bleu'],
                      val_out=config['val_dev_out'],
                      val_best_out=config['val_best_dev_out'],
                      every_n_batches=config['bleu_val_freq'],
                      bleu_out=config['val_bleu_out']))

if config['bleu_script'] is not None and config['train_val']:
    # compute bleu score for a small subset of the training data
    dev_stream = get_dev_stream(val_set=config['src_data_sample'], **config)
    extensions.append(
        BleuEvaluator(sampling_input, samples=samples, config=config,
                      model=search_model, data_stream=dev_stream,
                      ground_truth=config['trg_data_sample'],
                      normalize=config['normalized_bleu'],
                      val_out=config['val_train_out'],
                      val_best_out=config['val_best_train_out'],
                      every_n_batches=config['bleu_val_freq']))


fixed_params = []
if config['fix_embeddings']:
    fixed_params += [decoder.sequence_generator.readout.feedback_brick.lookup.W, encoder.lookup.W]

parameters = [param for param in cg.parameters if param not in fixed_params]

# Reload model if necessary
if config['load_weights']:
    extensions.append(LoadParameters(config['load_weights']))
elif config['reload']:
    extensions.append(LoadNMT(config['saveto']))

# Set up training algorithm
logger.info("Initializing training algorithm")
algorithm = GradientDescent(
    cost=cost, parameters=parameters,
    step_rule=CompositeRule([StepClipping(config['step_clipping']),
                             AdaDelta()])
)

# Initialize main loop
logger.info("Initializing main loop")
main_loop = MainLoop(
    model=training_model,
    algorithm=algorithm,
    data_stream=tr_stream,
    extensions=extensions
)

# Train!
main_loop.run()
