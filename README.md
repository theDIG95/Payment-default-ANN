# Payment-default-ANN

A generic ANN Classifier used for classification problem of 'payment default of credit cards dataset'. Available [here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).  
Made using the Theano framework in python

## Usage

### Same dataset

Download the data and adjust `DATA_FILE_PATH` in `data_importer.py` accordingly.

### Other datasets

The `ANNClassifier` can be used for any classification problem. The parameter's can be passed in the constructor and `fit`/`k_fold_x_validation` method for training.  
Weights can be saved and loaded by `save_weights` and `load_weights` methods.
Predictions can be made via the `predict` method.  
For further details please view the `ann.py` file, it contains docstrings and comments explaining each step of the process.

## Model properties

A sequential neural network with the following implementations.

- Batched gradient descent.
- Nestrov's momentum to improve learning with batches.
- RMSProp for adaptive learning rate.
- Dropout regularization.
- K-Fold cross validation.

## Model Parameters

For the 'payment default of credit cards dataset'

- Two hidden layer with depths (30, 20)
- Learning rate (0.005)
- Layer's probability of keeping (1, 1, 1)
- RMSProp cache decay (0.9)
- Nestrov's momentum (0.9)
- Epochs (1,000)
- Test/validation split (3) for 1 part validation (20,000 vs 10,000)
- Batch size (250)

## Results

Output of training

```(bash)
100  iterations,  20.69 % train error...
200  iterations,  19.39 % train error...
300  iterations,  18.91 % train error...
400  iterations,  19.45 % train error...
500  iterations,  18.775 % train error...
600  iterations,  18.655 % train error...
700  iterations,  18.34 % train error...
800  iterations,  18.529999999999998 % train error...
900  iterations,  18.615000000000002 % train error...
Training final error:  18.224999999999998 %
Testing final error:  17.740000000000002 %
```
