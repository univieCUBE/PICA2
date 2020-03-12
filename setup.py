#!/usr/bin/env python


import codecs
from os import path
import re
from setuptools import setup, find_namespace_packages
from pkg_resources import parse_requirements


# Single-sourcing the package version: Read from __init__
def read(*parts):
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements/prod.txt') as prod_req:
    requirements = [str(ir) for ir in parse_requirements(prod_req)]
with open('requirements/test.txt') as test_req:
    test_requirements = [str(tr) for tr in parse_requirements(test_req)]
with open('requirements/extras.txt') as extra_req:
    extra_requirements = [str(tr) for tr in parse_requirements(extra_req)]

setup(
    author="Lukas Lüftinger",
    author_email='lukas.lueftinger@outlook.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    description="Microbial Phenotype Prediction",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='phenotrex',
    name='phenotrex',
    entry_points={'console_scripts': [
        'phenotrex = phenotrex.cli.main:main',
    ], },
    packages=find_namespace_packages(),
    test_suite='tests',
    tests_require=requirements + test_requirements,
    extras_require={
        'fasta': extra_requirements
    },
    url='https://github.com/univieCUBE/phenotrex',
    version=find_version("phenotrex", "__init__.py"),  # update there
    zip_safe=False,
)
