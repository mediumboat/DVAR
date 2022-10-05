# DVAR

## 1. Description for each file
	DVAR_data_loader.py : the example codes to load the raw data and make it in a proper format for our model.
	DVAR_model.py : the implementation of DVAR model and training steps.
	DVAR_train.py : the example codes for training and testing the model.
	parameter.py : the hyper-parameters used in our model.

## 2. Requirements (Environment)
	python
	tensorflow
  	numpy
    scipy
    numpy_indexed
  	pandas
  	tqdm 
    networkx
    csrgraph


## 3. How to run

- (1) Configure hyper-parameters in parameter.py
- (2) Run "python DVAR_data_loader.py" for data preprocess.
- (3) Run "python DVAR_train.py".



## 4. Datasets

The datasets could be downloaded from the links in the paper (a sample data is provided in the repository), readers should also modify the "DVAR_data_loader.py" if they use their custom dataset. Please refer to the sample data for the input format.
