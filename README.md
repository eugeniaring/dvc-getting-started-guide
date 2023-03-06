# Getting started with DVC

![](https://github.com/eugeniaring/dvc-getting-started-guide/blob/main/datapipeline.png)

In this tutorial, we are going to master the principal commands and create a data pipeline with DVC.


## Detailed description of the project

The article with the explanations is [Getting started with DVC]()


## Tools used in the project

* [DVC](https://dvc.org/)
* [Catboost](https://catboost.ai/)

## Project Structure

* ```data/```: contains all the data
    * ```raw_data/```: contains original data
    * ```processed_data/```: contained processed data
* ```src```: contains the following scripts
    * ```preprocess.py```: Python script to preprocess the dataset
    * ```split.py```: Python script to split pre-processed data into training and test sets
    * ```train.py```: Python script to train catboost model, save artifact and performances
