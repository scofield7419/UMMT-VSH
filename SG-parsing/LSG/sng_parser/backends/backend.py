#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__all__ = ['ParserBackend']


class ParserBackend(object):
    """
    Based class for all parser backends. This class
    specifies the methods that should be override by subclasses.
    """

    def parse(self, sentence):
        raise NotImplementedError()

