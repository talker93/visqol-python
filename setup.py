from setuptools import setup, find_packages

setup(
    name="visqol",
    version="3.3.3",
    description="ViSQOL - Virtual Speech Quality Objective Listener (Pure Python)",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Google LLC, Andrew Hines (Python port)",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "soundfile>=0.10",
    ],
    entry_points={
        "console_scripts": [
            "visqol=visqol.__main__:main",
        ],
    },
    package_data={
        "visqol": ["model/*"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
)
