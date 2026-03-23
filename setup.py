from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="visqol-python",
    version="3.3.3",
    description="ViSQOL - Virtual Speech Quality Objective Listener (Pure Python)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/talker93/visqol-python",
    author="Shan Jiang",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "soundfile>=0.10",
        "libsvm-official>=3.25",
    ],
    entry_points={
        "console_scripts": [
            "visqol=visqol.__main__:main",
        ],
    },
    package_data={
        "visqol": ["model/*.txt"],
    },
    include_package_data=True,
    keywords=[
        "audio-quality", "speech-quality", "MOS", "PESQ", "POLQA",
        "visqol", "objective-metric", "perceptual-quality",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering",
    ],
    project_urls={
        "Bug Reports": "https://github.com/talker93/visqol-python/issues",
        "Source": "https://github.com/talker93/visqol-python",
        "Original C++": "https://github.com/google/visqol",
    },
)
