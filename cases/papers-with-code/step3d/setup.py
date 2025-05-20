#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="step3d",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.19.0",
        "PyQt5>=5.15.0",
        "PyQt5-sip>=12.8.0",
        "PyQt5-Qt5>=5.15.0",
        "OCC-Core>=7.5.0",
        "pythonocc-core>=7.5.0",
    ],
    entry_points={
        "console_scripts": [
            "step-viewer=step3d.ui.run_viewer:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for viewing and analyzing STEP files",
    keywords="CAD, STEP, 3D",
    python_requires=">=3.6",
)
