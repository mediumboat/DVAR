# Code repository for DVAR

## 1. Description for each file
	DVAR_data_loader.py : the example codes to load the raw data and make it in a proper format for our model.
	DVAR_model.py : the implementation of DVAR model and training steps.
	DVAR_train.py : the example codes for training and testing the model.
	parameter.py : the hyper-parameters used in our model.
	Other scripts are forked from metapath2vec repository.
	

## 2. Requirements (Environment)
	python >= 3.4.0
	tensorflow >= 2.2.0
  	numpy
    scipy
    numpy_indexed
  	pandas
  	tqdm 
    networkx
    csrgraph


## 3. How to run

- (1) Configure hyper-parameters in parameter.py
- (2) Run `python DVAR_data_loader.py` for data preprocess.
- (3) Run `python DVAR_train.py`.



## 4. Datasets

The datasets could be downloaded from the links in the paper (a sample data is provided in the repository), readers should also modify the "DVAR_data_loader.py" if they use their custom dataset. Please refer to the sample data for the input format.


## 5. Other Instructions

This model utilize `metapath2vec` for feature initiliazation, readers can also have their own implementation to produce node features (i.e., ` node_embeddings` in DVAR_train.py). For any enquiries, please contact zzliu[DOT]2020[AT]phdcs[DOT]smu[DOT]edu[DOT]sg
