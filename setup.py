#!/usr/bin/env python
from pathlib import Path
import subprocess as sp

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='simple_track',
    description='Simple cloud tracking',
    license='LICENSE',
    packages=[
        'simple_track',
    ],
)
