# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='rigidpy',
    version='0.0.1',
    description='Python package for rigidity analysis',
    long_description=readme,
    author='Mahdi Sadjadi, Varda Faghir Hagh',
    author_email='vfaghirh@asu.edu',
    url='https://github.com/vfaghirh/Rigidpy',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
