#! /usr/bin/python3

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from data_importer import synthetic_data, get_data

class HiddenLayer():
    def __init__(self, features, depth, pkeep, layer_no):
        """A single instance of a hidden layer
        
        Arguments:
            features {int} -- Number of input features to the layer
            depth {int} -- Depth of the layer
            pkeep {float} -- Layer's probability of keeping nodes/neurons
            layer_no {int} -- Layer number (for debugging)
        """

        # Layer input features
        self._input_feat = features
        # Layer depth
        self.depth = depth
        # Weight initial random values
        self._w_init = np.random.randn(self._input_feat, self.depth) / np.sqrt(self._input_feat + self.depth)
        # Updatable weights
        self.weights = theano.shared(self._w_init, 'W_{}'.format(layer_no))
        # Layer's initial cache for RMSProp
        self._cache = 0
        # Initial Nestrov's velocities
        self._nest_v = 0
        # Dropout keeping probability
        self.pkeep = pkeep

    def layer_update_eq(self, cost_fn, learning_rate, decay_rate, eps, mu):
        """Returns the layer's update equation for training the weights
        
        Arguments:
            cost_fn -- The loss/cost function fo the model
            learning_rate {float} -- Learning rate for gradient descent
            decay_rate {float} -- Cache decay rate for RMSProp
            eps {float} -- Epsilon value for RMSProp
            mu {float} -- Value for Nestrov's momentum
        
        Returns:
            The update equation for the layer's weights
        """

        # Calculate the gradient for layer
        grad = T.grad(cost_fn, self.weights)
        # Update cache
        self._cache = (decay_rate * self._cache) + ((1 - decay_rate) * grad * grad)
        # Update Nestrov's velocities
        self._nest_v = (mu * self._nest_v) - (learning_rate * grad)
        # Update the weights
        return (
            self.weights + (mu * self._nest_v - ((learning_rate * grad) / T.sqrt(self._cache + eps)))
        )

    def layer_output(self, input_to_layer):
        """Defines the output of the hidden layer
        
        Arguments:
            input_to_layer {Theano variable} -- Input to the layer
        
        Returns:
            The output of the layer including the activation function
        """

        # Dot product of input and weights
        return (
            T.tanh(input_to_layer.dot(self.weights))
            #T.nnet.relu(input_to_layer.dot(self.weights))
        )

class FinalLayer():
    def __init__(self, features, classes, pkeep):
        """The final output layer of the neural network
        
        Arguments:
            features {int} -- Number of input features to the layer
            classes {int} -- Number of output classes
            pkeep {float} -- Layer's probability of keeping nodes/neurons
        """

        # Layer input features
        self._input_feat = features
        # number of output classes
        self._classes = classes
        # Weight initial random values
        self._w_init = np.random.randn(self._input_feat, self._classes) / np.sqrt(self._input_feat + self._classes)
        # Updatable weights
        self.weights = theano.shared(self._w_init, 'W_Final')
        # Layer's initial cache for RMSProp
        self._cache = 0
        # Initial Nestrov's velocities
        self._nest_v = 0
        # Dropout keeping probability
        self.pkeep = pkeep

    def layer_update_eq(self, cost_fn, learning_rate, decay_rate, eps, mu):
        """Returns the layer's update equation for training the weights
        
        Arguments:
            cost_fn -- The loss/cost function fo the model
            learning_rate {float} -- Learning rate for gradient descent
            decay_rate {float} -- Cache decay rate for RMSProp
            eps {float} -- Epsilon value for RMSProp
            mu {float} -- Value for Nestrov's momentum
        
        Returns:
            The update equation for the layer's weights
        """

        # Calculate the gradient for layer
        grad = T.grad(cost_fn, self.weights)
        # Update cache
        self._cache = (decay_rate * self._cache) + ((1 - decay_rate) * grad * grad)
        # Update Nestrov's velocities
        self._nest_v = (mu * self._nest_v) - (learning_rate * grad)
        # Update the weights
        return (
            self.weights + (mu * self._nest_v - ((learning_rate * grad) / T.sqrt(self._cache + eps)))
        )

    def layer_output(self, input_to_layer):
        """Defines the output of the hidden layer
        
        Arguments:
            input_to_layer {Theano variable} -- Input to the layer
        
        Returns:
            The output of the layer including the activation function
        """
        # Dot product of input and weights
        return (
            T.nnet.softmax(input_to_layer.dot(self.weights))
        )

class ANNClassifier():
    def __init__(self, learning_rate, hidden_layer_depths, layers_pkeep, cache_decay_rate, nestrov_mu):
        """Artificial neural network classifier
        
        Arguments:
            learning_rate {float} -- Learning rate for Gradient descent
            hidden_layer_depths {list} -- Depths of each layer, excluding final layer
            layers_pkeep {list} -- Probability of keeping nodes/neurons for each layer, including input layer
            cache_decay_rate {float} -- Cache decay rate for RMSProp
            nestrov_mu {float} -- Nestrov's momentum value
        """

        # Learning rate
        self._lr = learning_rate
        # Number of hidden layers
        self._no_hl = len(hidden_layer_depths)
        # Depth of hidden layers
        self._hl_depths = hidden_layer_depths
        # Reference to hidden layers
        self._h_layers = None
        # Inputs and targets as symbols
        self._x_sym = T.matrix('X')
        self._t_sym = T.matrix('T')
        # Output placeholder
        self._y = None
        # Prediction classes placeholder
        self._pred = None
        # Final output layer
        self._f_layer = None
        # Cost function expression
        self._cost_fn = None
        # Training function placeholder
        self._train_fn = None
        # Prediction function placeholder
        self._pred_fn = None
        # Cache decay rate
        self._cahce_decay = cache_decay_rate
        # Epsilon for RMSProp
        self._eps =  1e-10
        # Mu for Nestrov's momentum
        self._mu = nestrov_mu
        # Probability of keeping for dropout regularization
        self._pkeeps = layers_pkeep
        # Theano random stream for droput mask
        self._rand_stream = RandomStreams()

    def _init_train_model(self, inputs, targets):
        """[INTERNAL] Initialize the model for training and setup Theano's functions
        
        Arguments:
            inputs {numpy.ndarray} -- Inputs to the model
            targets {numpy.ndarray} -- Target labels
        """

        # Create an empty list of hidden layers
        self._h_layers = list()
        # Create hidden layers
        for l_no in range(self._no_hl):
            # For first layer
            if l_no == 0:
                layer = HiddenLayer(inputs.shape[1], self._hl_depths[l_no], self._pkeeps[l_no], l_no)
            #For remaining middle layers
            else:
                layer = HiddenLayer(self._hl_depths[l_no - 1], self._hl_depths[l_no], self._pkeeps[l_no], l_no)
            self._h_layers.append(layer)
        # Create activation layer
        self._f_layer = FinalLayer(self._hl_depths[-1], targets.shape[1], self._pkeeps[-1])
        # Define outputs of hidden layers
        layer_z = self._x_sym
        for layer in self._h_layers:
            mask = self._rand_stream.binomial(n=1, p=layer.pkeep, size=layer_z.shape)
            layer_z = mask * layer_z
            layer_z = layer.layer_output(layer_z)
        # Define output of activation layer
        mask = self._rand_stream.binomial(n=1, p=layer.pkeep, size=layer_z.shape)
        layer_z = mask * layer_z
        self._y = self._f_layer.layer_output(layer_z)
        # Define cost function
        self._cost_fn = -(self._t_sym * T.log(self._y)).sum()
        # Predictions
        self._pred = T.argmax(self._y, axis=1)
        # Define tainting function
        self._train_fn = theano.function(
            inputs=[self._x_sym, self._t_sym],
            updates=[
                pair for pair in zip(
                    [layer.weights for layer in self._h_layers],
                    [layer.layer_update_eq(self._cost_fn, self._lr, self._cahce_decay, self._eps, self._mu) for layer in self._h_layers]
                )
            ] + [(self._f_layer.weights, self._f_layer.layer_update_eq(self._cost_fn, self._lr, self._cahce_decay, self._eps, self._mu)), ],
            allow_input_downcast=True
        )
        # Define training prediction function
        self._pred_fn = theano.function(
            inputs=[self._x_sym, self._t_sym],
            outputs=[
                self._cost_fn, self._pred
            ],
            allow_input_downcast=True
        )

    def _set_prediction_model(self):
        """[INTERNAL] Set the model for predicting labels for inputs
        """

        # Redefine outputs of hidden layers
        layer_z = self._x_sym
        for layer in self._h_layers:
            layer_z = layer.layer_output(layer_z * layer.pkeep)
        # Redefine output of activation layer
        self._y = self._f_layer.layer_output(layer_z * self._h_layers[-1].pkeep)
        # Redefine predictions
        self._pred = T.argmax(self._y, axis=1)
        # Redefine prediction function
        self._pred_fn = theano.function(
            inputs=[self._x_sym, ],
            outputs=[self._pred, ],
            allow_input_downcast=True
        )

    def _calc_error(self, targets, predictions):
        """[INTERNAL] Calculate error rate of predictions wrt target labels
        
        Arguments:
            targets {numpy.ndarray} -- Target labels
            predictions {numpy.ndarray} -- Model's predictions
        """

        # Remove extra dimensions form target matrix if necessary
        if targets.shape != predictions.shape:
            targets = np.squeeze(targets)
        # Calculate error of predictions
        return(np.mean(targets != predictions))

    def _separate_test_train(self, inputs, targets, parts_train):
        """[INTERNAL] Separate data into testing and training sets by random selection
        
        Arguments:
            inputs {numpy.ndarray} -- Inputs to the model
            targets {numpy.ndarray} -- target labels
            parts_train {int} -- Parts used for training vs 1 part for testing
        
        Returns:
            Tuple -- Two tuples of training inputs/labels and testing inputs/labels
        """

        # Find length of training data
        train_length = int(inputs.shape[0] - (inputs.shape[0] / parts_train))
        # Select random indices for train data
        indices = np.random.choice(
            np.arange(inputs.shape[0]), size=train_length, replace=False
        )
        # Separate training data
        inputs_train = inputs[indices, :]
        targets_train = targets[indices, :]
        # Separate test data
        inputs_test = np.delete(inputs, indices, axis=0)
        targets_test = np.delete(targets, indices, axis=0)
        # Make tuples for train and test data
        train_data = (inputs_train, targets_train)
        test_data = (inputs_test, targets_test)
        return (train_data, test_data)

    def _targets_to_ind_mat(self, targets):
        """[INTERNAL] Convert 1D target labels array to indicator matrix
        
        Arguments:
            targets {numpy.ndarray} -- Target labels
        
        Returns:
            numpy.ndarray -- Indicator matrix
        """

        # Initialize indicator matrix
        ind_mat = np.zeros(
            [
                targets.shape[0],
                int(np.amax(targets)) + 1
            ]
        )
        # Fill columns
        for i in range(targets.shape[0]):
            ind_mat[i, int(targets[i])] = 1
        return ind_mat

    def _plot_training(self, costs, err_rates):
        """[INTERNAL] Plot the training costs and error rates over all iterations
        
        Arguments:
            costs {list} -- Costs per iteration for training
            err_rates {list} -- Error rates per iteration for training
        """

        # Plot training costs
        plt.plot(costs)
        plt.xlabel('Iterations')
        plt.ylabel('Costs')
        plt.title('Training Costs over all iterations')
        plt.show()
        # Plot training errors
        plt.plot(err_rates)
        plt.xlabel('Iterations')
        plt.ylabel('Error rates')
        plt.title('Error rate of predictions (range 0-1) over all iterations')
        plt.show()

    def _confusion_targets_to_labels(self, targets, labels):
        """[INTERNAL] Convert numerical labels to string labels
        
        Arguments:
            targets {numpy.ndarray} -- Numerical target labels
            labels {list} -- String target labels
        
        Returns:
            list -- String labels
        """

        labels_list = list()
        for tgt in targets:
            labels_list.append(labels[int(tgt)])
        return labels_list

    def _plot_confusion_matrix(self, targets, predicted, labels=None):
        """[INTERNAL] Plot the confusion matrix for testing data
        
        Arguments:
            targets {numpy.ndarray} -- Target labels
            predicted {numpy.ndarray} -- Predicted labels
        
        Keyword Arguments:
            labels {list} -- String labels for numerical targets (default: {None})
        """

        # Numerical targets to labels
        targets_lbl = self._confusion_targets_to_labels(targets, labels)
        # Numerical predictions to labels
        preds_lbl = self._confusion_targets_to_labels(predicted, labels)
        # get confusion matrix
        cm = confusion_matrix(targets_lbl, preds_lbl, labels=labels)
        # Normalize confusion matrix
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Show confusion matrix
        plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
        plt.title('Confusion matrix for testing data')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        if labels:
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=45)
            plt.yticks(tick_marks, labels)
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        plt.show()

    def add_input_bias_col(self, inputs):
        """Add bias column to inputs to absorb weights
        
        Arguments:
            inputs {numpy.ndarray} -- Inputs to the model
        
        Returns:
            numpy.ndarray -- Inputs with bias column
        """

        # Check inputs shape
        if len(inputs.shape) <= 1:
            inputs = np.expand_dims(inputs, axis=1)
        # Column of ones to accommodate bias into weights
        bias_col = np.ones([inputs.shape[0], 1])
        return np.hstack([bias_col, inputs])

    def save_weights(self, file_path):
        """Save the trained weights to file
        
        Arguments:
            file_path {str} -- Path of the save file
        """

        with open(file_path, mode='w') as save_file:
            for layer in self._h_layers:
                w_vals = layer.weights.get_value()
                for val in w_vals.flatten():
                    save_file.write(str(val) + ',')
                save_file.write('\n' + str(w_vals.shape[0]) + ',' + str(w_vals.shape[1]) + '\n')
            f_w_vals = self._f_layer.weights.get_value()
            for val in f_w_vals.flatten():
                save_file.write(str(val) + ',')
            save_file.write('\n' + str(f_w_vals.shape[0]) + ',' + str(f_w_vals.shape[1]) + '\n')

    def load_weights(self, file_path, inputs, targets):
        """Load weights from a saved file
        
        Arguments:
            file_path {str} -- Path of saved file
            inputs {numpy.ndarray} -- Inputs to the model
            targets {numpy.ndarray} -- Target labels
        """

        # Initialize the model
        self._init_train_model(inputs, targets)
        # Placeholder for weights
        weights_list = list()
        # Read the weights from file
        with open(file_path, mode='r') as load_file:
            for line in load_file:
                # Parse weights
                weights = line.split(',')
                weights = list(map(str.strip, weights))
                for weight in weights:
                    try:
                        float(weight)
                    except:
                        weights.remove(weight)
                weights = np.array(list(map(float, weights)))
                # Read shape from file
                shape = load_file.readline()
                # Parse shape
                shape = shape.split(',')
                shape = list(map(str.strip, shape))
                shape = list(map(int, shape))
                # Reshape weights
                weights = np.reshape(weights, [shape[0], shape[1]])
                # Add final weights to list
                weights_list.append(weights)
        # Set weight of final layer
        self._f_layer.weights = theano.shared(weights_list[-1], 'W')
        # Set weights of hidden layers
        for l_no, layer in enumerate(self._h_layers):
            layer.weights = theano.shared(weights_list[l_no], 'W')

    def fit(self, inputs, targets, no_iters, parts_train, batch_size, thresh=1e-10, thresh_limit=100, plot_errs=False, confusion_labels=False):
        """Fit the model to the given data
        
        Arguments:
            inputs {numpy.ndarray} -- Inputs to train to
            targets {numpy.ndarray} -- Target labels
            no_iters {int} -- Number of iterations to train for/epochs
            parts_train {int} -- Parts of inputs/targets for training vs 1 part for validation
            batch_size {int} -- Batch size for training
        
        Keyword Arguments:
            thresh {int} -- Error percentage change which will break the iterations if met (default: {1e-10})
            thresh_limit {int} -- Number of continuos iterations for which the error threshold has to be met (default: {100})
            plot_errs {bool} -- Plot training costs and error rates (default: {False})
            confusion_labels {list(string)} -- If given, will plot the confusion matrix for the testing part (default: {False})
        """

        # Iteration break threshold counter
        thresh_count = 0
        # Separate training and testing data
        train_data, test_data = self._separate_test_train(inputs, targets, parts_train)
        inputs_train = train_data[0]
        targets_train = train_data[1]
        inputs_test = test_data[0]
        targets_test = test_data[1]
        # Targets to indicator matrix
        train_ind = self._targets_to_ind_mat(targets_train)
        # Initialize the model
        self._init_train_model(inputs_train, train_ind)
        # Placeholders for costs and error rates
        costs = list()
        err_rates = list()
        # Shuffle the data
        inputs_train_shuff, targets_train_shuff = shuffle(inputs_train, train_ind)
        # Calculate number of batches
        no_batches = int(inputs_train.shape[0] / batch_size)
        # Current batch number
        batch_no = 0
        # For specified iterations
        for i in range(no_iters):
            # Train the model
            self._train_fn(
                inputs_train_shuff[(batch_no*batch_size) : (batch_no*batch_size)+batch_size],
                targets_train_shuff[(batch_no*batch_size) : (batch_no*batch_size)+batch_size]
            )
            # Find cost and error rate
            cost, train_pred = self._pred_fn(inputs_train, train_ind)
            costs.append(float(cost))
            if np.isnan(costs[-1]):
                print('WARNING!!! NaN cost after {} iterations, breaking...'.format(i))
                break
            err_rates.append(self._calc_error(targets_train, train_pred))
            # Print progress of training
            if (not i == 0) and ((i % 100) == 0):
                _, temp_pred = self._pred_fn(inputs_test, targets_test)
                temp_err = self._calc_error(targets_test, temp_pred)
                print(i, ' iterations |', round(err_rates[-1]*100, 3), '% train error', end=' | ')
                print(round(temp_err*100, 3), '% testing error ...')
            # Increment batch number
            batch_no += 1
            # Check batch number for overflow
            if batch_no >= no_batches:
                batch_no = 0
            # Check error change for threshold
            if len(err_rates) > 1:
                if np.sqrt(np.square(err_rates[-2] - err_rates[-1]))*100 <= thresh:
                    thresh_count += 1
                    # Break if threshold met
                    if thresh_count >= thresh_limit:
                        print('Test error threshold met after {} continuos iterations, breaking iterations...'.format(i))
                        break
                else:
                    thresh_count = 0
        # Predict for test data
        test_pred = self.predict(inputs_test)
        test_err = self._calc_error(targets_test, test_pred)
        # Print Results
        print('Training final error: ', err_rates[-1]*100, '%')
        print('Testing final error: ', test_err*100, '%')
        # Plot training progress if needed
        if plot_errs:
            self._plot_training(costs, err_rates)
        # Plot testing confusion matrix if needed
        if confusion_labels:
            self._plot_confusion_matrix(targets_test, test_pred, confusion_labels)
        # Return costs and error rates
        return(costs, err_rates, test_err)

    def predict(self, inputs):
        """Predict label(s) for the given inputs
        
        Arguments:
            inputs {numpy.ndarray} -- Inputs for which prediction is required
        
        Returns:
            numpy.ndarray -- Predicted labels
        """

        # Redefine model for prediction
        self._set_prediction_model()
        # Make predictions
        preds = self._pred_fn(inputs)
        # Convert to numpy array if necessary
        if type(preds) == list:
            preds = np.squeeze(np.array(preds))
        return preds

    def k_fold_x_validation(self, inputs, targets, no_iters, parts_train, batch_size, params, thresh=1e-10, thresh_limit=100):
        """K-Fold Cross validations for set of model parameters
        
        Arguments:
            inputs {numpy.ndarray} -- Inputs to train to
            targets {numpy.ndarray} -- Target labels
            no_iters {int} -- Number of iterations to train for/epochs
            parts_train {int} -- Parts of inputs/targets for training vs 1 part for validation
            batch_size {int} -- Batch size for training
            params {list of tuples} -- Parameters to test for (same order as for __init__)
        
        Keyword Arguments:
            thresh {int} -- Error percentage change which will break the iterations if met (default: {1e-10})
            thresh_limit {int} -- Number of continuos iterations for which the error threshold has to be met (default: {100})
        """

        # Placeholder for test errors
        test_errs = list()
        # For each parameter set
        for pram in params:
            print('=======================================')
            print('For parameters: ', pram)
            # Set model parameters
            self._lr = pram[0]
            self._no_hl = pram[1]
            self._hl_depths = pram[2]
            self._pkeeps = pram[3]
            self._cahce_decay = pram[4]
            self._mu = pram[5]
            # Train the model
            _, _, test_err = self.fit(inputs, targets, no_iters, parts_train, batch_size, thresh=thresh, thresh_limit=thresh_limit)
            # Store the errors
            test_errs.append(test_err)
        # Print results
        print('=======================================')
        print('Lowest testing error is for parameter set number: ', test_errs.index(min(test_errs)) + 1)

def main():
    # Synthetic data testing
    '''print('Getting data...')
    inputs, targets = synthetic_data()
    print('Initializing model...')
    model = ANNClassifier(0.005, [3, 2], [1.0, 1.0, 1.0], 0.9, 0.9)
    inputs = model.add_input_bias_col(inputs)
    print('Fitting model to data...')
    model.fit(inputs, targets, 1000, 10, 60, plot_errs=True, confusion_labels=['c1', 'c2', 'c3', 'c4'])'''

    # Credit data
    print('Getting data...')
    inputs, targets = get_data()
    print('Initializing model...')
    model = ANNClassifier(0.005, [30, 20], [1.0, 1.0, 1.0], 0.9, 0.9)
    inputs = model.add_input_bias_col(inputs)
    print('Fitting model to data...')
    model.fit(inputs, targets, 1000, 3, 250, plot_errs=True, confusion_labels=['No', 'Yes'])

if __name__ == '__main__':
    main()
