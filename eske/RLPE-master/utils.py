#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import pickle
import codecs


class AttrDict(dict):
    """
    Dictionary whose elements can be accessed like attributes.

    >>> d = AttrDict(x=1, y=2)
    >>> d.x = 2
    d => {'x': 2, 'y': 2}
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        # Pure magic
        self.__dict__ = self

    def setdefault(self, *args, **kwargs):
        """
        Set default values for the given keys.
        The entries whose key is already present in the dictionary are not modified.
        Modify the dictionary in-place, and return a reference to self.

        >>> d = AttrDict(x=1, y=2)
        >>> d.setdefault(y=3, z=3)
        >>> {'x': 2, 'y': 2, 'z': 3}

        'z' does not exist and is inserted,
        'y' already exists, it is not modified.

        This method is still compatible with dict.setdefault:
        >>> d = AttrDict(x=1, y=2)
        >>> d.setdefault('z', 4)
        >>> 4
        >>> d.setdefault('y', 3)
        >>> 2
        """
        if args:
            # For retro-compatibility with dict
            return super(AttrDict, self).setdefault(*args)
        for k, v in kwargs.items():
            super(AttrDict, self).setdefault(k, v)
        return self


def uopen(filename, mode='r', encoding='utf-8'):
    return codecs.open(filename, mode, encoding)


def uprint(unicode_text):
    print unicode_text.encode('utf-8')


def pickle_dump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = list(range(len(s1) + 1))

    for j, y in enumerate(s2):
        new_distances = [j + 1]
        for i, x in enumerate(s1):
            if x == y:
                new_distances.append(distances[i])
            else:
                new_distances.append(1 + min((distances[i],
                                              distances[i + 1],
                                              new_distances[-1])))
        distances = new_distances

    return distances[-1]
