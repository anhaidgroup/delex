## How to Install Delex on a Linux Machine

This is a step-by-step guide to install Delex on a single Linux machine. This guide has been tested on a Linux machine running Ubuntu 22.04 with 8 GB memory, using Python 3.10, Java Temurin 17 JDK, and PyLucene 9.4.1. You can try to adapt this guide to other configurations.

### Step 1: Installing Sparkly

Sparkly is a blocking-for-entity-matching package that Delex is built on top of. So you must install Sparkly using [this guide](https://github.com/anhaidgroup/sparkly/blob/main/doc/install-single-machine-linux.md). 

### Step 2: Installing Delex

Now you can install Delex. To do so, you can pip install from PyPI or from GitHub. The default is to pip install from PyPI. You may want to pip install from GitHub if you want to install the latest Delex version compared to the version on PyPI. For example, the GitHub version may contain bug fixes that the PyPI version does not.

#### Option 1: Pip Installing from PyPI

You can install Delex from PyPI using the following command:

	pip install delex

This command will install Delex and the following dependencies: Joblib, Matplotlib, Networkx, Numba, Numpy, Pandas, Py\_Stringmatching, Pyarrow, Pydantic, Pydot PySpark, Scipy, Sparkly, Tqdm, and xxhash. These are all of Delex's dependencies, except Java, JCC, and PyLucene.

Java, JCC and PyLucene cannot be pip installed with the above command, because they are not available on PyPI. If you followed the above Sparkly installation instructions, then you have already installed Java, JCC, and PyLucene as a part of the Sparkly installation. 

#### Option 2: Pip Installing from GitHub

To install Delex directly from its GitHub repo, use the following command:

	pip install https://github.com/anhaidgroup/delex.git

Similar to pip installing from PyPI, the above command will install Delex and all of its dependencies, except Java, JCC, and PyLucene (which have been installed as a part of the Sparkly installation process). 


