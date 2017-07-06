#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function


from setuptools import setup

setup(
    name = "ink",
    version = "0.1",
    author = "Vsevolod Svetlov",
    author_email = "svetlov.vsevolod@gmail.com",
    packages=["ink"],
    entry_points={
        'console_scripts': [
            "ink = ink.ink:main"
        ]
    }
)
