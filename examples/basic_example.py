import sys
sys.path.append('.')
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pathlib import Path

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

# enable pyarrow execution, recommended for better performance
conf = SparkConf()\
        .set('spark.sql.execution.arrow.pyspark.enabled',  'true')

# initialize a local spark context
spark = SparkSession.builder\
                    .master('local[*]')\
                    .config(conf=conf)\
                    .appName('Basic Example')\
                    .getOrCreate()

# path to the test data
data_path = (Path().resolve()).absolute()
# table to be indexed
index_table_path = data_path / 'table_a.parquet'
# table for searching
search_table_path = data_path / 'table_b.parquet'
# the ground truth
gold_path = data_path / 'gold.parquet'

# read all the data as spark dataframes
index_table = spark.read.parquet(f'file://{str(index_table_path)}')
search_table = spark.read.parquet(f'file://{str(search_table_path)}')
gold = spark.read.parquet(f'file://{str(gold_path)}')
index_table.printSchema()

prog = BlockingProgram(
        keep_rules = [
                KeepRule([ JaccardPredicate('title', 'title', QGramTokenizer(3), operator.ge, .6)])
            ],
        drop_rules = [],
    )


executor = PlanExecutor(
        index_table=index_table, 
        search_table=search_table,
        optimize=False,
        estimate_cost=False,
)

candidates, stats = executor.execute(prog, ['_id'])
candidates = candidates.persist()

candidates.show()
pairs = candidates.select(
                    F.explode('ids').alias('a_id'),
                    F.col('_id').alias('b_id')
                )
gold = gold.drop('__index_level_0__')
n_pairs = pairs.count()
true_positives = gold.intersect(pairs).count()
recall = true_positives / gold.count()
print(f'n_pairs : {n_pairs}')
print(f'true_positives : {true_positives}')
print(f'recall : {recall}')

candidates.unpersist()

