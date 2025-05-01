# Step-by-step guide to running Delex

This guide is a step-by-step guide to running Delex. For this guide, we will assume that you have already installed everything from the provided [Cloud Installation Guide](https://github.com/anhaidgroup/delex/blob/docs/doc/installation-guides/install-cloud-based-cluster.md) .

## Step One: Download datasets --- We should change when the datasets are hosted to make it easier using wget

To begin, we need to download the datasets from the GitHub. Navigate to the dblp_acm folder here: https://github.com/anhaidgroup/active_matcher/tree/main/examples/data/dblp_acm. Then, click on 'gold.parquet' and click the download icon at the top. Repeat this for ''table_a.parquet',and 'table_b.parquet'. Now, using your file manager on your computer, move these all into one folder called 'dblp_acm'. (if the data is not hosted somewhere, we will need to add instructions about using scp). This should be done on all of the nodes.

## Step Two: Create Python file

On the master node, in the 'dblp_acm' directory, create a file called 'example.py'. We will use this Python file to walkthrough the code.

## Step Three: Import dependencies

Now, we can open up the 'example.py' file. Before we begin, we first need to import all of the necessary packages that we will use.

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

## Step Four: Initialize Spark

Next we need to initialize Spark. For this example we are doing everything in a cloud setup. All files need to be stored in the ~/dblp_acm directory on all nodes.

```
# enable pyarrow execution, recommended for better performance
conf = SparkConf()\
        .set('spark.sql.execution.arrow.pyspark.enabled',  'true')

# initialize a local spark context
spark = SparkSession.builder\
                    .master('{url of spark master}')\
                    .config(conf=conf)\
                    .appName('Basic Example')\
                    .getOrCreate()
```

### Data

The data we downloaded contains files in parquet format. This is a small dataset of paper citations with about 1000 rows per table.

```
# path to the test data directory
data_path = Path('/home/ubuntu/dblp_acm')

# table to be indexed, generally this should be the table with fewer rows
index_table_path = data_path / 'table_a.parquet'
# table for searching
search_table_path = data_path / 'table_b.parquet'
# the ground truth, i.e. the correct matching pairs
gold_path = data_path / 'gold.parquet'
```

## Step Five: Read the Data

Once Spark is initialized, we can then read all of our data into Spark dataframes.

```
# read all the data as spark dataframes
index_table = spark.read.parquet(f'file://{str(index_table_path)}')
search_table = spark.read.parquet(f'file://{str(search_table_path)}')
gold = spark.read.parquet(f'file://{str(gold_path)}')

index_table.printSchema()
```

## Step Six: Create a Blocking Program

Next we need to define our blocking program. For this basic example, we will define a very simple blocking program that returns all pairs where the Jaccard scores using a 3gram tokenizer are greater than or equal to .6. To do this we define a BlockingProgram with a single KeepRule which has a single JaccardPredicate.

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

## Step Seven: Execute a Blocking Program

Next, we create a PlanExecutor and execute the BlockingProgram by calling .execute(). Notice, that we passed optimize=False and estimate_cost=False as arguments, these parameters control the plan that is generated, which will be explained in a separate example.

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

## Step Eight: Compute Recall

Finally, we can compute recall. As you can see, the output of the PlanExecutor is actually grouped by the id of search_table. This is done for space and computation effeicency reasons. To compute recall we first need to 'unroll' the output and then do a set intersection with the gold pairs to get the number of true positives.

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

## Step Nine: Running on a Cluster

In order to run this on a cluster, we can use the following command from the root directory (you can always get to the root directory by typing `cd` into the terminal). **Note**: This command assumes that the directory structure is the same as ours, and if you followed our installation guides, it will be the same.

```
spark/bin/spark-submit \
  --master {url of Spark Master} \
  /home/ubuntu/dblp_acm/example.py
```
