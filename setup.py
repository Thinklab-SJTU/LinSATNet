#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree
import re

from setuptools import find_packages, setup, Command

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)


# Package meta-data.
NAME = 'LinSATNet'
DESCRIPTION = 'LinSATNet offers a neural network layer to enforce the satisfiability of positive linear constraints ' \
              'to the output of neural networks. The gradient through the layer is exactly computed. This package ' \
              'now works with PyTorch.'
URL = 'https://github.com/Thinklab-SJTU/LinSATNet'
AUTHOR = get_property('__author__', NAME)
VERSION = get_property('__version__', NAME)

REQUIRED = [
     'torch>=1.0.0'
]

EXTRAS = {}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
# The contents in `skip_sections` will be ingored.
skip_sections = ['### Table of contents',
                 '## How it works?',
                 '## More Complicated Use Cases (appeared in our paper)']
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n'
        skip_sec_level = -1
        for line in f.readlines():
            if line.strip() in skip_sections:
                skip_sec_level = line.count('#')
            elif skip_sec_level > 0:
                if line[0] == '#':
                    cur_sec_level = line.count('#')
                    if cur_sec_level <= skip_sec_level:
                        skip_sec_level = -1
            if skip_sec_level < 0:
                long_description += line
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=(
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Environment :: GPU :: NVIDIA CUDA',
        'Environment :: Console',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ),
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
