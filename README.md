# Getting started with DVC

![](https://github.com/eugeniaring/dvc-getting-started-guide/blob/main/datapipeline.png)

In this tutorial, we are going to master the principal commands and create a data pipeline with DVC.


## Detailed description of the project

The article with the explanations is [Getting started with DVC]()

## Project Structure

* ```data/```: contains all the data
    * ```raw_data/```: contains original data
    * ```processed_data/```: contained processed data
* ```src```: contains the following scripts
    * ```preprocess.py```: Python script to preprocess the dataset
    * ```split.py```: Python script to split pre-processed data into training and test sets
    * ```train.py```: Python script to train catboost model, save artifact and performances

## Set up the project 

1. Clone the template branch:

```git clone --branch template https://github.com/eugeniaring/dvc-getting-started-guide.git```

2. Create a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

3. Install requirements

```
pip install -r requirements.txt
```
