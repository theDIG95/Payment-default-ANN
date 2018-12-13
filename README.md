# Payment-default-ANN

ANN Classifier for the payment default of credit cards dataset. Available [here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

## Model

A sequential neural network with the following implementations.

- Batched gradient descent.
- Nestrov's momentum to improve learning over batches.
- RMSProp for adaptive learning rate.
- Dropout regularization.

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

## Model Parameters

- Two hidden layer with depths (30, 20)
- Learning rate (0.005)
- Layer's probability of keeping (1, 1, 1)
- RMSProp cache decay (0.9)
- Nestrov's momentum (0.9)
- Epochs (1,000)
- Test/validation split (3) for 1 part validation
- Batch size (250)
