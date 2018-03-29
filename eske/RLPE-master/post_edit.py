#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from operator import itemgetter
from random import shuffle
from sklearn.tree import ExtraTreeRegressor


def training(transitions, v, phi, get_actions, gamma):
    """
    :param transitions: list of pairs (state, reward)
    :param v: value function in the current iteration
    :param phi: function returning the vector representation of a state
    :param get_actions: function returning the available actions in a given state
    :param gamma: discount factor
    :return: training set used to update the value function for the next iteration
    """
    training_set = []
    for s, r in transitions:
        actions = get_actions(s)
        shuffle(actions)
        o = r + gamma * max(v(phi(s.transition(a))) for a in actions)
        training_set.append((phi(s), o))

    return training_set


def post_edit(state, v, phi, get_actions, max_horizon):
    value = v(phi(state))

    transitions = []

    for _ in range(max_horizon):
        actions = get_actions(state)
        shuffle(actions)
        states = [state.transition(a) for a in actions]
        values = [v(phi(s)) for s in states]

        new_value, new_action, new_state = max(zip(values, actions, states), key=itemgetter(0))  # greedy policy
        if new_value <= value:       # when no state has a better value than current state, stop
            new_action = ('STOP',)
            new_state = state

        transitions.append((new_value, new_action, new_state))
        value, action, state = new_value, new_action, new_state

        if action == ('STOP',):
            break

    return transitions, state


def get_regressor(training_set):
    """
    Estimation of the value function using a regression algorithm.
    Training set contains tuples of (state, score)
    V: S -> R
    """
    clf = ExtraTreeRegressor()
    clf.fit(*zip(*training_set))
    return clf


def get_value_function(regressor):
    if regressor is None:
        return lambda x: 0
    else:
        return lambda x: regressor.predict(x)[0]
