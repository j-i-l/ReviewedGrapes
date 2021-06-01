#!/usr/bin/env python

import os
from distutils.core import setup

GITHUB_LOCATION = "https://github.com/j-i-l/reviewedgrapes"


def get_files(directory):
    """Collect all files for the trained models."""
    files = []
    for r, d, f in os.walk(directory):
        for af in f:
            tf = os.path.join(r, af)
            files.append(os.path.join(*(tf.split(os.path.sep)[1:])))
    return files


setup(name='ReviewedGrapes',
      version='1.0',
      description='Distribution of fitted ML models that predict wine variety '
                  'based on a wine review text',
      author='Jonas I Liechti',
      author_email='j-i-l@t4d.ch',
      # url='https://www.python.org/sigs/distutils-sig/',
      packages=['reviewed_grapes'],
      package_dir={'reviewed_grapes': 'reviewed_grapes'},
      package_data={'reviewed_grapes':
                    get_files('reviewed_grapes/fitted_models')},
      # data_files=get_files('reviewed_grapes/fitted_models'),
      license='GPLv3',
      install_requires=['nltk>=3.6.2',
                        'pyspark>=3.1.1',
                        'numpy>=1.19.5'],
      classifiers=['Intended Audience :: Science/Research',
                   'Development Status :: 5 - Production/Stable',
                   'License :: OSI Approved :: GPLv3 License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python'])
