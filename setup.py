# Copyright 2016-2020 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-retinamask/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

from codecs import open

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, 'README.md'), 'r', 'utf-8') as f:
    readme = f.read()


about = {}
with open(os.path.join(here, 'deepcell_retinamask', '_version.py'), 'r', 'utf-8') as f:
    exec(f.read(), about)


setup(
    name=about['__title__'],
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    description=about['__description__'],
    url=about['__url__'],
    download_url=about['__download_url__'],
    license=about['__license__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=[
        'deepcell~=0.9.0',
        'deepcell-toolbox~=0.9.0',
        'numpy>=1.16.6,<1.20.0',
        'opencv-python-headless<5',
        'tensorflow~=2.4.1',
        'scikit-image>=0.14.5'
    ],
    extras_require={
        'tests': [
            'pytest<6',
            'pytest-cov',
            'pytest-pep8',
        ],
    },
    packages=find_packages(),
    python_requires='>=3.6, <3.9',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
