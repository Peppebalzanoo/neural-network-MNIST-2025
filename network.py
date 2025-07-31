import numpy as np
import activation as act


class Network:
    def __init__(self, input_n: int, hiddens_n: list, output_n: int, sigma: float = 0.001):
        """Initialize the neural network with given parameters."""
        if input_n is None or output_n is None:
            raise ValueError("Input and output neurons must be specified.")
        if input_n <= 0 or output_n <= 0:
            raise ValueError("Input and output neurons must be positive integers.")
        if not isinstance(hiddens_n, list) or len(hiddens_n) == 0:
            raise ValueError("Hidden neurons must be a non-empty list.")
        if any(n <= 0 for n in hiddens_n):
            raise ValueError("All hidden neurons must be positive integers.")

        self._input_neurons: int = input_n  # Number of input neurons
        self._hidden_neurons: list = hiddens_n  # List of hidden layer neurons
        self._output_neurons: int = output_n  # Number of output neurons
        self._depth: int = 0  # Depth of the network (number of layers excluding input layer)

        self._sigma: float = sigma  # Standard deviation for weight initialization
        self._biases: list = []  # List of biases for each layer
        self._weights: list = []  # List of weights for each layer

        self._act_functions: list = []  # List of activation functions for each layer

        self._init_params()  # Initialize parameters: biases, weights, and activation functions

    def _init_params(self) -> None:
        """Initialize biases, weights, and activation functions for the network."""
        prev_conn_num: int = self._input_neurons
        for curr_hidden in self._hidden_neurons:
            # Biases array for each curr_hidden neurons (one bias for each neuron)
            self._biases.append(self._sigma * np.random.normal(size=[curr_hidden, 1]))
            # Weights matrix for each curr_hidden neurons
            self._weights.append(self._sigma * np.random.normal(size=[curr_hidden, prev_conn_num]))
            # Set activation function for each hidden layer
            self._act_functions.append(act.tanh)

            prev_conn_num = curr_hidden

        # Biases array for each output neurons (one bias for each neuron)
        self._biases.append(self._sigma * np.random.normal(size=[self._output_neurons, 1]))
        # Weights matrix for each output neurons
        self._weights.append(self._sigma * np.random.normal(size=[self._output_neurons, prev_conn_num]))
        # Set activation function for output layer
        self._act_functions.append(act.identity)

        # Calculate the depth of the network (excluding input layer)
        self._depth = len(self._weights)

    def copy_network(self) -> "Network":
        """Create a copy of the current network instance."""
        new_net = Network(self._input_neurons, self._hidden_neurons, self._output_neurons, self._sigma)
        new_net._biases = [np.copy(b) for b in self._biases]
        new_net._weights = [np.copy(w) for w in self._weights]
        new_net._act_functions = list(self._act_functions)
        return new_net

    def get_accuracy(self, Z, Y) -> float:
        """Calculate the accuracy of the network."""
        # X are the network predictions, Y are the one-hot labels
        # Both have shape (number_samples, number_classes)
        total_labels = Y.shape[0]
        predicted_labels = np.argmax(Z, axis=1)
        true_labels = np.argmax(Y, axis=1)
        correct = np.sum(predicted_labels == true_labels)
        return correct / total_labels

    def forward_propagation(self, X) -> np.ndarray:
        """Perform forward propagation through the network."""
        for layer in range(0, self._depth):
            # print(f"Layer {layer}: newX.shape: {X.shape}")
            R = np.matmul(X, self._weights[layer].T) + self._biases[layer].T
            X = self._act_functions[layer](R)
        return X

    def _forward_prop_train(self, X) -> tuple:
        """Perform forward propagation during training."""
        Y_act_values = [X]  # List of outputs for each layer using activation functions
        Y_act_derivatives = []  # List of derivatives of activation functions for each layer
        for layer in range(0, self._depth):
            R = np.matmul(Y_act_values[layer], self._weights[layer].T) + self._biases[layer].T
            Y_act_values.append(self._act_functions[layer](R))
            curr_act_fun = self._act_functions[layer]
            Y_act_derivatives.append(curr_act_fun(R, derivative=True))
        return Y_act_values, Y_act_derivatives

    def _back_propagation(self, X_train, Y_train, error_function, eta):
        """Train the network on the provided dataset."""
        # Step 1: Forward Step
        Y_act_values, Y_act_derivatives = self._forward_prop_train(X_train)
        # Step 2: Compute the error for each layer (delta values)
        delta_values = []
        for layer in range(self._depth, 0, -1):  # Iterate from output layer to input layer
            if layer == self._depth:
                # Calculate the error for the output layer
                delta_k = error_function(Y_act_values[layer], Y_train, derivative=True) * Y_act_derivatives[layer - 1]
                # Append the delta for the output layer
                delta_values.append(delta_k)
            else:
                # Calculate the error for hidden layers
                delta_h = np.matmul(delta_values[0], self._weights[layer]) * Y_act_derivatives[layer - 1]
                # Append the delta for the hidden layer
                delta_values.insert(0, delta_h)

        # Step 3: Calculate all partial derivatives (local rows)
        der_partials = []
        for layer in range(0, self._depth):
            der_partial = np.matmul(delta_values[layer].T, Y_act_values[layer])
            der_partials.append(der_partial)

        # STEP 4: Update weights and biases
        self._gradient_descent(der_partials, delta_values, eta)

    def _gradient_descent(self, der_partials, delta_values, eta):
        """Update weights and biases using gradient descent."""
        # Get the number of samples in the batch
        num_samples = delta_values[0].shape[0]
        for layer in range(0, self._depth):
            # Update weights and biases with normalization
            self._weights[layer] -= (eta / num_samples) * der_partials[layer]
            # Bias update: correctly calculated from deltas and averaged
            bias_gradient = np.sum(delta_values[layer], axis=0, keepdims=True).T
            self._biases[layer] -= (eta / num_samples) * bias_gradient

    def fit(self, X_train, Y_train, X_valid, Y_valid, error_function, epoch_number=10, eta=0.1, patience=5) -> None:
        # Perform forward propagation and calculate the error
        Z_train = self.forward_propagation(X_train)
        error_train = error_function(Z_train, Y_train)

        # Perform forward propagation for validation set
        Z_valid = self.forward_propagation(X_valid)
        error_valid = error_function(Z_valid, Y_valid)

        # Print initial training information
        self._print_train_info(Z_train, Y_train, Z_valid, Y_valid, error_train, error_valid, -1)

        # Early stopping variables
        best_valid_error = float("inf")
        patience_counter = 0
        curr_epoca = 0
        while curr_epoca < epoch_number:
            # Perform back propagation
            self._back_propagation(X_train, Y_train, error_function, eta)

            # Perform forward propagation and calculate the error
            Z_train = self.forward_propagation(X_train)
            error_train = error_function(Z_train, Y_train)
            # Perform forward propagation for validation set
            Z_valid = self.forward_propagation(X_valid)
            error_valid = error_function(Z_valid, Y_valid)

            # Print training information
            self._print_train_info(Z_train, Y_train, Z_valid, Y_valid, error_train, error_valid, curr_epoca)

            curr_epoca += 1

            # Early stopping check
            if error_valid < best_valid_error:
                best_valid_error = error_valid
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {curr_epoca} due to no improvement in validation error.")
                    break

    def get_info(self) -> None:
        """Print the architecture and parameters of the network."""
        print(f"\nNetwork Architecture: {self._input_neurons}  x {self._hidden_neurons} x {self._output_neurons}")
        print(f"Weights Shapes: {[w.shape for w in self._weights]}")
        print(f"Biases Shapes: {[b.shape for b in self._biases]}")
        print(f"Activation Functions: {[act.__name__ for act in self._act_functions]}")

    def _print_train_info(self, Z_train, Y_train, Z_valid, Y_valid, error_train, error_valid, curr_epoca):
        print(
            f"Epoca: {curr_epoca}, Train error: {error_train:.15f}, Accuracy Train: {self.get_accuracy(Z_train, Y_train):.15f}, Validation error: {error_valid:.15f}, Accuracy Validation: {self.get_accuracy(Z_valid, Y_valid):.15f}"
        )
