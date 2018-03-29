#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import chain
from operator import itemgetter
from collections import defaultdict
from utils import uopen
import numpy as np

"""
Extract trajectories/alignments from IBM alignment file, or TERCOM alignment

Trajectories are sequences of transitions:
    state, new state, action
Outputs a transition per line, the first number is the index of the trajectory the transition belongs to.
The fields of a transition are separated by '|||'.

Actions are:
        SUB index word
        DEL index
        INS index word
        MOVE index position
"""


class Action(object):
    # TODO
    SUB, DEL, INS, MOVE, STOP = range(5)
    def __init__(self, op):
        self.op = op


class State(object):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg

    def __iter__(self):
        yield self.src
        yield self.trg

    def transition(self, action):
        """
        Returns the resulting state after applying the given action in the given state.
        """
        trg = self.trg.split()
        op = action[0]
        if op == 'SUB':
            trg[action[1]] = action[2]
        elif op == 'DEL':
            del trg[action[1]]
        elif op == 'INS':
            trg.insert(action[1], action[2])
        elif op == 'MOVE':
            trg.insert(action[2], trg.pop(action[1]))
        elif op == 'STOP':
            pass
        else:
            raise Exception('unknown action')

        return State(self.src, ' '.join(trg))


class Trajectories(object):
    """
    Lists trajectories from a batch training file. Iteration is possible either on trajectories, or
    on the transitions.
    """

    def __init__(self, filename):
        self.trajectories = self.read_trajectories(filename)
        self.seen_actions = self.read_actions()
        self.reward_parameters = None

    @staticmethod
    def read_trajectories(filename):
        """
        A trajectory file contains entries each corresponding to a transition, whose
         fields are delimited by '|||'.
         Transitions contain those fields: id, src, s, s', a, score(s), score(s')
         The first field is the id of this transition's trajectory.
        """
        trajectories = []

        with uopen(filename) as f:
            current_idx = None
            current_traj = []
            for line in f:
                idx, src, trg1, trg2, action, score1, score2 = line.split('|||')
                score1, score2 = float(score1), float(score2)

                action = action.split()
                op = action[0]
                if op in ['SUB', 'INS', 'DEL']:
                    action[1] = int(action[1])
                elif op == 'MOVE':
                    action[1:] = map(int, action[1:])
                elif op != 'STOP':
                    raise Exception('Unknown action type')

                action = tuple(action)

                # a state is a pair (source sentence, translation hypothesis)
                # source sentence is the same for all transitions in a given trajectory
                state1 = State(src, trg1)
                state2 = State(src, trg2)
                transition = (state1, state2, action, score1, score2)

                # a different index marks the end of a trajectory
                if current_idx is not None and idx != current_idx:
                    trajectories.append(current_traj)
                    current_traj = []

                current_traj.append(transition)
                current_idx = idx

        return trajectories

    def compute_scores(self, phi, gamma):
        """
        Use score-based inverse reinforcement learning (SBIRL) to estimate a reward function from
        the trajectories and their scores.
        The resulting reward function is a linear combination of the feature function:
            r(s) = theta . phi(s)

        The learning is done by assuming that there exists a theta such that for each trajectory h = (s0, ..., sT)
        and its score V(h):
            V(h) = r(s_0) + gamma * r(s_1) + ... + gamma^T * r(s_T)

        theta is estimated by minimizing the least square error on the training set.
        """
        # TODO: score is TER or 1-TER?
        features = []
        scores = []

        for trajectory in self.trajectories:
            v = trajectory[0][3]

            mu = 0
            for i, transition in enumerate(trajectory):  # include last transition?
                state1, state2, _, _, _ = transition

                #phi1 = phi(state1)
                phi2 = phi(state2)

                if phi2 is not None:
                    mu += gamma ** i * phi2

                #if phi1 is not None and phi2 is not None:
                    #mu += gamma ** i * (phi2 - phi1)
                    #mu += gamma ** i * np.concatenate([phi1, phi2])

            #states = map(itemgetter(0), trajectory) + [trajectory[-1][1]]
            #mu = 0
            #for i, state in enumerate(states):
            #    mu += gamma ** i * phi(state)

            features.append(mu)
            scores.append(v)

        outer = sum(np.outer(mu, mu) for mu in features)
        inv = np.linalg.inv(outer)

        theta = np.linalg.inv(inv).dot(sum(mu * v for mu, v in zip(features, scores)))
        #theta = np.linalg.inv(sum(np.outer(mu, mu) for mu in features)).dot(
        #    sum(mu * v for mu, v in zip(features, scores)))

        self.reward_parameters = theta

    def SBIRL(self, phi):
        for trajectory in self.trajectories:
            for _, transition in enumerate(trajectory):
                state1, state2, action, _, _ = transition
                # TODO: reward depends only on s or s'? Doesn't make sense... Does it?
                # TODO: phi(s,s') = phi(s') - phi(s) and r(s,s') = theta . phi(s,s') would make better sense
                phi2 = phi(state2)
                if phi2 is not None:
                    reward = np.dot(self.reward_parameters, phi2)
                    #reward = np.dot(self.reward_parameters, np.concatenate([phi(state1), phi(state2)]))
                    yield state1, state2, action, reward

    def read_actions(self):
        """
        Retrieves the actions seen in the training corpus.
        Returns a dictionary containing words or word pairs for SUB, INS, DEL and MOVE operations.
        """
        seen_actions = defaultdict(set)

        # counts for each kind of action
        insertions = defaultdict(int)
        substitutions = defaultdict(int)
        moves = defaultdict(int)

        for state, _, action, _, _ in self.__iter__():
            trg = state.trg
            words = trg.split()

            op = action[0]
            if op == 'SUB':
                word_pair = words[action[1]], action[2]
                substitutions[word_pair] += 1
            elif op == 'INS':
                word = action[2]
                insertions[word] += 1
            elif op == 'DEL':
                pos = action[1]
                word = words[pos]
                seen_actions[op].add(word)
            elif op == 'MOVE':
                word = words[action[1]]
                moves[word] += 1

        # 100 most frequent inserted words
        for word, _ in sorted(insertions.items(), key=itemgetter(1), reverse=True)[:100]:
            seen_actions['INS'].add(word)

        # desperate effort to reduce the number of operations
        min_count = 2

        for word_pair, count in substitutions.items():
            if count >= min_count:
                seen_actions['SUB'].add(word_pair)

        for word, count in moves.items():
            if count >= min_count:
                seen_actions['MOVE'].add(word)

        return seen_actions

    def __iter__(self):
        return chain(*self.trajectories)

    def reward_shaping(self, gamma):
        """
        Yields transitions with a reward instead of a distinct score for each state.
        The reward is computed using the "reward shaping" method:
            r(s, s') = gamma * score(s')                  if s = s0
            r(s, s') = gamma * score(s') - score(s)       otherwise
        """
        for trajectory in self.trajectories:
            for i, transition in enumerate(trajectory):
                state1, state2, action, score1, score2 = transition
                reward = gamma * score2 - score1 if i != 0 else gamma * score2
                yield state1, state2, action, reward

    def get_actions(self, state):
        """
        Returns all available actions from a given state (a sentence).
        Actions are:
            MOVE pos pos
            SUB pos word
            DEL pos
            INS pos word
        Restricts actions to those seen in the training corpus.
        """
        restrict_move = True
        restrict_del = False

        actions = []
        trg = state.trg.split()

        for pos, word in enumerate(trg):
            # DEL
            if not restrict_del or word in self.seen_actions['DEL']:
                actions.append(('DEL', pos))

            # SUB
            for w1, w2 in self.seen_actions['SUB']:
                if w1 == word:
                    actions.append(('SUB', pos, w2))

            # INS
            for ins_word in self.seen_actions['INS']:
                actions.append(('INS', pos, ins_word))
                # Allow insertion in last position
                if pos == len(trg) - 1:
                    actions.append(('INS', len(trg), ins_word))

            # MOVE
            if not restrict_move or word in self.seen_actions['MOVE']:
                for move_pos in range(len(trg)):
                    actions.append(('MOVE', pos, move_pos))

        actions.append(('STOP',))

        return actions
