#!/usr/bin/env python3
"""
Setup script for the sentence transformer fine-tuning package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sentence-transformer-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready sentence transformer fine-tuning for binary text classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/sentence-transformer-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    # No console scripts - use the CLI files directly (./train, ./evaluate, ./publish)
    include_package_data=True,
    package_data={
        "src": ["configs/*.yaml"],
    },
)
