# How to Install Delex on a Linux Machine

This is a step-by-step guide to install Delex and its necessary dependencies on a single Linux machine.

This guide has been tested on a Linux Machine running Ubuntu 22.04 with 8 GB memory. The following software versions were installed on the test machine using the steps in this guide: Python 3.10, Java Temurin 17 JDK, and PyLucene 9.4.1. You can try to adapt this guide to other configurations, but we have not tested it and cannot guarantee that it will work.

## Step 1: Sparkly and Sparkly Dependency Installation

Sparkly is an entity matching package that Delex is built on top of. Therefore, Delex requires Sparkly as a dependency, so we need to install Sparkly (and Sparkly’s dependencies) first. To download Sparkly, and its dependencies, please refer to the step-by-step guide here: [Sparkly Linux Installation](https://github.com/anhaidgroup/sparkly/blob/docs-update/doc/install-single-machine-linux.md).

Once you have completed the guide above, you can install Delex. 

## Step 2: Delex Installation

To download Delex, you can use one of the following options: pip installing from PyPI or pip installing from GitHub. You may want to pip install from GitHub if you want to install the latest Delex version compared to the version on PyPI. For example, the GitHub version may contain bug fixes that the PyPI version does not.

### Option 1: Pip installing from PyPI

You can install Sparkly from PyPI, using the following command:

	pip install delex

This command will install Delex and the following dependencies: Joblib, Matplotlib, Networkx, Numba, Numpy, Pandas, Py\_Stringmatching, Pyarrow, Pydantic, Pydot PySpark, Scipy, Sparkly, Tqdm, and xxhash. These are all of the dependencies, except Java, JCC, and PyLucene.

Java, JCC and PyLucene cannot be pip installed with the above command, because they are not available on PyPI. If you followed the Sparkly installation instructions, then you have installed Java, JCC, and PyLucene already.

### Option 2: Pip Installing from GitHub

To install Delex directly from its GitHub repo, use the following command:

	pip install https://github.com/anhaidgroup/delex.git

Similar to pip installing from PyPI, the above command will install Delex and all of its dependencies, except Java, JCC, and PyLucene.


