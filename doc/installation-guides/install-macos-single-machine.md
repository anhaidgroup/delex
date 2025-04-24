# How to Install Delex on a MacOS M1 Machine

This is a step-by-step guide to install Delex and its necessary dependencies on a single macOS machine with an M1 chip. If you are unsure if your Mac has an M1 chip, click on the Apple in the top left corner of your screen \> About This Mac. If it says Chip Apple M1, then you have an M1 chip. If it does not say Chip Apple M1, you do not have an M1 chip.

This guide has been tested on a 2020 MacBook Pro with an Apple M1 Chip, 8GB Memory, macOS version Sequoia 15.0.1, and a .zshrc profile. The following software versions were installed on the test machine using the steps in this guide: Python 3.12, ICU4C 74.2, Java Temurin 17 JDK, and PyLucene 9.12.0. You can try to adapt this guide to other configurations.

If your machine has an Intel chip, this installation guide will not work for you. If your machine has an M2, M3, or M4 chip, this installation guide may work for you, but we have not tested it, and we can not guarantee that it will work.

## Step 1: Sparkly and Sparkly Dependency Installation

Sparkly is an entity matching package that Delex is built on top of. Therefore, Delex requires Sparkly as a dependency, so we need to install Sparkly (and Sparkly’s dependencies) first. To download Sparkly, and its dependencies, please refer to the step-by-step guide here: [link](%20https://github.com/anhaidgroup/sparkly/blob/docs-update/doc/install-single-machine-macOS.md).

Once you have completed the guide above, you can download and install Delex. 

## Step 2: Delex Installation

To download and install Delex, you can use one of the following options: pip installing from PyPI or pip installing from GitHub. You may want to pip install from GitHub if you want to install the latest Delex version compared to the version on PyPI. For example, the GitHub version may contain bug fixes that the PyPI version does not.

Before installing, make sure you are in the correct virtual environment. This step is necessary because this is the environment where we installed all of the dependencies. To make sure the environment is activated, run the following:  
	source \~/sparkly/bin/activate

### Option 1: Pip installing from PyPI

You can install Sparkly from PyPI, using the following command:

	pip install delex

This command will install Delex and the following dependencies: Joblib, Matplotlib, Networkx, Numba, Numpy, Pandas, Py\_Stringmatching, Pyarrow, Pydantic, Pydot PySpark, Scipy, Sparkly, Tqdm, and xxhash. These are all of the dependencies, except Java, JCC, and PyLucene.

Java, JCC and PyLucene cannot be pip installed with the above command, because they are not available on PyPI. If you followed the Sparkly installation instructions, then you have installed Java, JCC, and PyLucene already.

### Option 2: Pip Installing from GitHub

To install Delex directly from its GitHub repo, use the following command:

	pip install https://github.com/anhaidgroup/delex.git

Similar to pip installing from PyPI, the above command will install Delex and all of its dependencies, except Java, JCC, and PyLucene.


