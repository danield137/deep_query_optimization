# deep_query_optimization
Code for my Thesis on Deep Query Execution Time Estimation

Work is organized into folders by topics:

## datasets
Contains the scripts needed to load the datasets used for training, as well as some wrappers using to load, iterate, clean and evalute the datasets. 
Datasets are comprized from source sql's (TPCH, TPCD, TPCDS, IMDB) and scripts to load them.  
It also contains the queries execution results.

## db
This contains the db client, extended with useful functions like stats collection for later stages.

## estimator
This contains various models for query estimation, most of which are implements as pytorch (lightning) models (model, dataset and training files).
Also contains evluation code and some snapshots of the trained models.

## lab
Contains code to generate query execution time datasets.

## query_generator
Contains various query executors (file based, online generated).

## relational
Methods for working with relational trees (serializing, deseriazling and other needed funcs).
