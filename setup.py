# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="rigidpy",
    version="0.0.8",
    description="rigidpy package for rigidity analysis",
    keywords="rigidity physics math python flexibility condensedmatter",
    author="Varda Faghir Hagh, Mahdi Sadjadi",
    author_email="vardahagh@uchicago.edu",
    url="https://github.com/vfaghirh/rigidpy",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.5",
)
