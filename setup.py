#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Setup file for PySparkStock.
"""

import sys
from setuptools import setup, find_packages


def setup_package():

    setup(
        name='PySparkStock',
        packages=find_packages(),
        description='PySparkStock',
        author='UIUC',
        zip_safe=False,
        include_package_data=True,
        root_script_source_version="python3.4",
        default_python="python3.4",
        install_requires=[
            'pandas',
            'boto3'
      ]
    )


if __name__ == "__main__":
    setup_package()
    