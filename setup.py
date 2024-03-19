# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from setuptools import find_packages, setup

KW = ["artificial intelligence", "deep learning", "unsupervised learning", "contrastive learning"]


EXTRA_REQUIREMENTS = {
    "dali": ["nvidia-dali-cuda110"],
    "umap": ["matplotlib", "seaborn", "pandas", "umap-learn"],
    "h5": ["h5py"],
}


def parse_requirements(path):
    with open(path) as f:
        requirements = [p.strip().split()[-1] for p in f.readlines()]
    return requirements


setup(
    name="solo-learn",
    packages=find_packages(exclude=["bash_files", "docs", "downstream", "tests", "zoo"]),
    version="1.0.6",
    license="MIT",
    author="solo-learn development team",
    author_email="vturrisi@gmail.com, enrico.fini@gmail.com",
    url="https://github.com/vturrisi/solo-learn",
    keywords=KW,
    install_requires=[
        "torch==1.10.0+cu113",
        "torchvision==0.11.1",
        "einops",
        "pytorch-lightning==1.6.4",
        "torchmetrics==0.6.0",
        "lightning-bolts>=0.5.0",
        "tqdm",
        "wandb",
        "scipy",
        "timm==0.6.13",
        "scikit-learn",
        "hydra-core",
        "setuptools==58.0.4",
    ],
    extras_require=EXTRA_REQUIREMENTS,
    dependency_links=["https://developer.download.nvidia.com/compute/redist"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)
