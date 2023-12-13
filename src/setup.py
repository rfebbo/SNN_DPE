#!/usr/bin/env python

from distutils.core import setup

setup(name='snn_dpe',
      version='0.1',
      description='Python Model for ',
      author='Rocco Febbo',
      author_email='febbo87@gmail.com',
      url='https://github.com/rfebbo/SNN_DPE',
      packages=['snn_dpe'],
      install_requires=[
          'numpy',
          'matplotlib',
          'ipykernel',
          'regex',
          'networkx',
          'tqdm',
          'scipy',
          'sympy'
      ],
     )
