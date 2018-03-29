#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import pdb


class Connection(object):
    def __init__(self, configuration=None):
        self.local = configuration is None

        if not self.local:
            self.configuration = configuration
            self.client = Client(self.configuration['client_configuration'],
                                 sshserver=self.configuration['controller'])
            self.view = client[:]
            self.n_processes = len(client.ids)

    def run(self, func):
        if self.local:
            func()
        else:
            self.view.apply_sync(func)

    def map(self, func, data):
        if self.local:
            return func(data)

        seg_length = 1 + len(data) // self.n_processes
        segments = [data[i * seg_length:(i + 1) * seg_length] for i in range(self.n_processes)]
        results = self.view.map_sync(func, segments)
        return sum(results, [])

    def set_globals(self, **kwargs):
        if self.local:
            globals().update(kwargs)    # FIXME: globals of this module...
        else:
            for k, v in kwargs:
                self.view[k] = v
