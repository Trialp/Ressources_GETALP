#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from random import shuffle
from post_edit import get_regressor
from remote import Connection
from utils import pickle_dump
import pdb
import os

datadir = 'data'
modeldir = 'models'
outputdir = 'output'
n_iterations = 1
gamma = 0.9
#client_conf = dict(controller='brahms1',
#                   client_configuration='/media/lig/.ipython/profile_default/security/ipcontroller-client.json')
client_conf = None

def load_data():
    from features import Model
    from trajectories import Trajectories
    from remote import *  # FIXME
    global model, phi, trajectories, get_actions, regressor
    model = Model(model_file)  # bivec model
    phi = model.paragraph_vector  # feature function: State -> vector
    trajectories = Trajectories(trajectory_file)
    get_actions = trajectories.get_actions


def training_(transitions):
    # globals required: regressor, phi, get_actions, gamma
    from post_edit import get_value_function, training
    from remote import *  # FIXME
    v = get_value_function(regressor)
    return training(transitions, v, phi, get_actions, gamma)


def main():
    model_file = os.path.join(modeldir, 'commoncrawl_fr-en.bin')
    trajectory_file = os.path.join(datadir, 'eolss-train.trajectories+scores.txt')

    con = Connection(configuration=client_conf)
    con.set_globals(trajectory_file=trajectory_file, model_file=model_file, gamma=gamma)
    con.run(load_data)

    trajectories.compute_scores(phi, gamma)
    transitions = [(s, r) for _, s, _, r in trajectories.SBIRL(phi)]  # in fitted-value iteration we care only about s'
    shuffle(transitions)

    regressor = None

    for k in range(n_iterations):
        print 'Iteration', k
        con.set_globals(regressor=regressor)

        training_set = con.map(training_, transitions)
        regressor = get_regressor(training_set)

        pickle_dump(regressor, 'output/regressor.{}.pickle'.format(k + 1))

if __name__ == '__main__':
    main()
