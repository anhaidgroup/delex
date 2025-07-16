## Running Delex on a Single Machine

Here we will walk through an example of running a simple blocking program for Delex on a single machine. In particular, we will show you how to create a Python program, step by step, then execute the Python program at the end of the walkthrough. You can see the complete Python program [here](./basic_example.py), and if you rather do the walkthrough using a pre-built Jupyter Notebook, you can do so [here](./basic_example.ipynb). 

We assume you have already installed Delex on a single machine, using [this guide](https://github.com/anhaidgroup/delex/blob/docs/doc/installation-guides/install-single-machine.md).

### Step 1: Downloading Datasets

To begin, we need to download three datasets from GitHub. Navigate to the dblp_acm folder [here](https://github.com/anhaidgroup/delex/tree/main/examples/data/dblp_acm). Click on 'gold.parquet' and click the download icon at the top. Repeat this for 'table_a.parquet' and 'table_b.parquet'. Now move all these into a directory on your local machine called 'dblp_acm'.

### Step 2: Creating a Python File

Within the 'dblp_acm' directory, create a Python file called 'basic_example.py'. We will use this Python file to write the code.

_Note: If you would like to see the completed Python file without following the walkthrough below, you can visit [basic_example.py](./basic_example.py). If you would like to complete the walkthrough using a pre-built Jupyter Notebook rather than your own Python file, you can visit [basic_example.ipynb](./basic_example.ipynb)._

### Step 3: Importing Dependencies

Now we can open up the 'basic_example.py' file, and add the following code that imports all of the necessary packages that we will use.

```
from pathlib import Path

import sys
sys.path.append(str(Path().resolve().parent))
import os
os.environ['PYTHONPATH'] = str(Path().resolve().parent)

from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


from delex.lang.predicate import (
        BM25TopkPredicate,
        JaccardPredicate,
        EditDistancePredicate,
        SmithWatermanPredicate,
        JaroPredicate,
        JaroWinklerPredicate,
        CosinePredicate,
        ExactMatchPredicate
)

from delex.lang import BlockingProgram, DropRule, KeepRule
from delex.tokenizer import StrippedWhiteSpaceTokenizer, QGramTokenizer
from delex.execution.plan_executor import PlanExecutor
import operator
import psutil
```

### Step 4: Initializing Spark

Next we initialize Spark. For this example we are doing everything in a local setup. In particular, all files are stored on the local file system and we are running Spark in the local mode.

```
# enable pyarrow execution, recommended for better performance
conf = SparkConf()\
        .set('spark.sql.execution.arrow.pyspark.enabled',  'true')

# initialize a local spark context
spark = SparkSession.builder\
                    .master('local[*]')\
                    .config(conf=conf)\
                    .appName('Basic Example')\
                    .getOrCreate()
```

#### Data

The data we downloaded earlier contains files in parquet format. This is a small dataset of paper citations with about 1000 rows per table.

```
# path to the test data directory
data_path = (Path().resolve()).absolute()

# table to be indexed, generally this should be the table with fewer rows
index_table_path = data_path / 'table_a.parquet'
# table for searching
search_table_path = data_path / 'table_b.parquet'
# the ground truth, i.e. the correct matching pairs
gold_path = data_path / 'gold.parquet'
```

### Step 5: Reading the Data

Once Spark has been initialized, we can read all of our data into Spark dataframes.

```
# read all the data as spark dataframes
index_table = spark.read.parquet(f'file://{str(index_table_path)}')
search_table = spark.read.parquet(f'file://{str(search_table_path)}')
gold = spark.read.parquet(f'file://{str(gold_path)}')

index_table.printSchema()
```

### Step 6: Creating a Blocking Program

Next, we define our blocking program. For this basic example, we will define a very simple blocking program that returns all pairs where the Jaccard scores using a 3gram tokenizer are greater than or equal to .6. To do this we define a BlockingProgram with a single KeepRule which has a single JaccardPredicate.

```
prog = BlockingProgram(
        keep_rules = [
                KeepRule([ JaccardPredicate('title', 'title', QGramTokenizer(3), operator.ge, .6)])
            ],
        drop_rules = [],
    )
```

In terms of SQL this program is equivalent to

```
SELECT A.id, B.id
FROM index_table as A, search_table as B
WHERE jaccard_3gram(A.title, B.title) >= .6
```

### Step 7: Executing the Blocking Program

Next, we create a PlanExecutor and execute the BlockingProgram by calling .execute(). Note that we passed optimize=False and estimate_cost=False as arguments. These parameters control the plan that is generated, which will be explained in a separate example.

```
executor = PlanExecutor(
        index_table=index_table,
        search_table=search_table,
        optimize=False,
        estimate_cost=False,
)

candidates, stats = executor.execute(prog, ['_id'])
candidates = candidates.persist()
candidates.show()
```

### Step 8: Computing Recall

Finally, we can compute the recall of the blocking step (that is, the fraction of true matches that survive blocking). As you can see, the output of the PlanExecutor is actually grouped by the id of search_table. This is done for space and computation efficiency reasons. To compute recall we first need to 'unroll' the output and then do a set intersection with the gold pairs to get the number of true positives.

```
# unroll the output
pairs = candidates.select(
                    F.explode('ids').alias('a_id'),
                    F.col('_id').alias('b_id')
                )
# total number
n_pairs = pairs.count()
true_positives = gold.intersect(pairs).count()
recall = true_positives / gold.count()
print(f'n_pairs : {n_pairs}')
print(f'true_positives : {true_positives}')
print(f'recall : {recall}')

# remove the dataframe from the cache
candidates.unpersist()
```

### Step 9: Running the Python Program

Congratulations! You have finished writing a Python script for blocking with Delex, and now you can run the script. To do so, open a terminal and navigate to the directory that you wrote your 'basic_example.py' file in. Finally, run the following command, and once the script is finished, it will output n_pairs, true_positives, and recall:

```
python3 basic_example.py
```
