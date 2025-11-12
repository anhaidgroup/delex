## How to Install Delex on a MacOS M1 Machine

This is a step-by-step guide to install Delex on a single macOS machine with an M1 chip. To check if your Mac has an M1 chip, click on the Apple in the top left corner of your screen \> About This Mac. If it says Chip Apple M1, then you have an M1 chip. If it does not say Chip Apple M1, you do not have an M1 chip.

This guide has been tested on a 2020 MacBook Pro with an Apple M1 Chip, 8GB Memory, macOS version Sequoia 15.0.1, and a .zshrc profile. The following software versions were installed on the test machine using the steps in this guide: Python 3.12, ICU4C 74.2, Java Temurin 17 JDK, and PyLucene 9.12.0. You can try to adapt this guide to other configurations.

If your machine has an Intel chip, this guide will not work for you. If your machine has an M2, M3, or M4 chip, this guide may work for you, but we have not tested it.

### Step 1: Installing Sparkly

Sparkly is a blocking-for-entity-matching package that Delex is built on top of. So you need to first install Sparkly, using [this guide](https://github.com/anhaidgroup/sparkly/blob/main/doc/install-single-machine-macOS.md).
 
### Step 2: Installing Delex

To install Delex, you can pip install from PyPI or pip install from GitHub. The default is to pip install from PyPI. However, you may want to pip install from GitHub if you want to install the latest Delex version compared to the version on PyPI. For example, the GitHub version may contain bug fixes that the PyPI version does not.

Before installing, make sure you are in the correct virtual environment. This step is necessary because this is the environment where you have installed all of the dependencies. If you followed the Sparkly installation instructions, you can activate the environment by running the following:
```
	source ~/sparkly-venv/bin/activate
```

Before installing Delex, we should also return to the root directory by running the following command in the terminal:

```
    cd
```

In the future you can install Delex using one of the following two options. **As of now, since Delex is still in testing, we do not yet enable Option 1 (Pip installing from PyPI). Thus you should use Option 2 (Pip installing from GitHub).**

#### Option 1: Pip Installing from PyPI

**Note that this option is not yet enabled. Please use Option 2.**

You can install Sparkly from PyPI using the following command:
```
	pip install delex
```
This command will install Delex and the following dependencies: Joblib, Matplotlib, Networkx, Numba, Numpy, Pandas, Py\_Stringmatching, Pyarrow, Pydantic, Pydot PySpark, Scipy, Sparkly, Tqdm, and xxhash. These are all of Delex's dependencies, except Java, JCC, and PyLucene.

Java, JCC and PyLucene cannot be pip installed with the above command, because they are not available on PyPI. If you followed the Sparkly installation instructions, then you have already installed Java, JCC, and PyLucene as a part of Sparkly installation process.

#### Option 2: Pip Installing from GitHub

To install Delex directly from its GitHub repo, use the following command:
```
	pip install git+https://github.com/anhaidgroup/delex.git@main
```
Similar to pip installing from PyPI, the above command will install Delex and all of its dependencies, except Java, JCC, and PyLucene (which must have been installed as a part of the Sparkly installation process). 


