## Delex: Combining Multiple Strategies for Blocking for Entity Matching

Delex is an open-source tool for the blocking step of entity matching. For more details on entity matching, see [this page](https://anhaidgroup.github.io/magellan/about). Delex is distinguished in the following aspects: 
* It enables using multiple strategies to perform blocking. 
* It uses Spark to scale to large tables, for example, with tens of millions or hundreds of millions of tuples per table. 

Delex is still in beta testing, and we are looking for users who want to use it (with active support from our team). 

### Motivation and Comparison with Sparkly

Initially we developed Sparkly, a simpler blocking solution that uses TF/IDF. We found that Sparkly outperforms many state-of-the-art blocking solutions. If you are looking for a solution to perform blocking, we highly recommend trying Sparkly first, to see if it is already sufficient for your problem. 

Sparkly uses only one blocking strategy, namely finding top-k candidate matches using the TF/IDF similarity score. While working with Sparkly, we observed that in some cases the user wants to use multiple blocking strategies. 

For example, the user may already have a dictionary-based blocking solution that the user has used for many years, and have also spent many years developing it. So the user may want to block using both the Sparkly blocker and this dictionary-based blocker. That is, the blocking step should output all tuple pairs that are output by the Sparkly blocker or the dictionary-based blocker.  
As another example, the user may write a blocking rule stating that if two tuples share similar names and zip codes, then they should be in the blocking output. Then the user wants to block using both the Sparkly blocker and this blocking rule. 

In addition to combining multiple blocking strategies, a user may also want to write rules to exclude certain tuple pairs from the blocking output, such as pairs where the zip codes do not match exactly. 

Delex enables all of the above. In particular, 
Delex provides a language that users can use to quickly write declarative blocking programs that combine multiple blocking strategies, as well as a variety of rules (such as rules to exclude certain tuple pairs from the blocking output). 
The above language allows users to plug in a wide variety of blackbox, pre-built blockers, such as a dictionary-based blocker that has been built over many years. 
Delex can translate and execute blocking programs written in the above language over a Spark cluster, thus scaling to large tables of hundreds of millions of tuples.  

How Delex Works

Given two tables A and B to be matched, Delex focuses on the blocking step. The user examines Tables A and B then writes a declarative blocking program that uses one or more blocking strategies. Delex translates this program into a DAG, optimizes the DAG, sends the optimized DAG as well as partitions of Tables A and B to the nodes in a Spark cluster, executes on the nodes, then returns the output. To execute fast on the Spark nodes, Delex examines the DAG to build a set of indexes, ships these indexes to the Spark nodes, then uses the indexes to execute. 

Implementation-wise, Delex uses Spark, Lucene, and Sparkly, among other software. 

Case Studies and Performance Statistics
Installation
How to Use
Further Pointers
. Delex is distinguished in the following aspects: 
It enables using multiple strategies to perform blocking. 
It uses Spark to scale to large tables, for example, with tens of millions or hundreds of millions of tuples per table. 

Delex is still in beta testing, and we are looking for users who want to use it (with active support from our team). 

### Motivation and Comparison with Sparkly

Initially we developed Sparkly, a simpler blocking solution that uses TF/IDF. We found that Sparkly outperforms many state-of-the-art blocking solutions. If you are looking for a solution to perform blocking, we highly recommend trying Sparkly first, to see if it is already sufficient for your problem. 

Sparkly uses only one blocking strategy, namely finding top-k candidate matches using the TF/IDF similarity score. While working with Sparkly, we observed that in some cases the user wants to use multiple blocking strategies. 

For example, the user may already have a dictionary-based blocking solution that the user has used for many years, and have also spent many years developing it. So the user may want to block using both the Sparkly blocker and this dictionary-based blocker. That is, the blocking step should output all tuple pairs that are output by the Sparkly blocker or the dictionary-based blocker.  
As another example, the user may write a blocking rule stating that if two tuples share similar names and zip codes, then they should be in the blocking output. Then the user wants to block using both the Sparkly blocker and this blocking rule. 

In addition to combining multiple blocking strategies, a user may also want to write rules to exclude certain tuple pairs from the blocking output, such as pairs where the zip codes do not match exactly. 

Delex enables all of the above. In particular, 
Delex provides a language that users can use to quickly write declarative blocking programs that combine multiple blocking strategies, as well as a variety of rules (such as rules to exclude certain tuple pairs from the blocking output). 
The above language allows users to plug in a wide variety of blackbox, pre-built blockers, such as a dictionary-based blocker that has been built over many years. 
Delex can translate and execute blocking programs written in the above language over a Spark cluster, thus scaling to large tables of hundreds of millions of tuples.  

### How Delex Works

Given two tables A and B to be matched, Delex focuses on the blocking step. The user examines Tables A and B then writes a declarative blocking program that uses one or more blocking strategies. Delex translates this program into a DAG, optimizes the DAG, sends the optimized DAG as well as partitions of Tables A and B to the nodes in a Spark cluster, executes on the nodes, then returns the output. To execute fast on the Spark nodes, Delex examines the DAG to build a set of indexes, ships these indexes to the Spark nodes, then uses the indexes to execute. 

Implementation-wise, Delex uses Spark, Lucene, and Sparkly, among other software. 

### Case Studies and Performance Statistics
### Installation
### How to Use
### Further Pointers
