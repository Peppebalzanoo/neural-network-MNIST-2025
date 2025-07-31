import numpy as np
from network import Network
from loader import DataLoader
from error import cross_entropy
import matplotlib.pyplot as plt


def kfold_cross_validation(loader: DataLoader, input_neurons, hidden_neurons, output_neurons, k, epoch_number, eta, patience) -> "Network":
    """Run k-fold cross-validation on the dataset."""

    validation_accuracies = []
    best_model = None

    # Create stratified folds
    fold_indices = loader.create_stratified_k_folds(loader._train_labels, k)

    # Iterate over each fold
    for i in range(0, k):
        print(f"\n--- Start Fold {i+1}/{k} ---")
        # Prepare the data for this specific fold
        val_indices = fold_indices[i]

        # The training set is composed of all other folds combined
        train_indices_list = [fold_indices[j] for j in range(k) if j != i]
        train_indices = np.concatenate(train_indices_list)

        # Select the data and labels using the indices
        if loader._train_data is None or loader._train_labels is None:
            raise ValueError("Training data or labels not loaded. Please call _load_data() first.")
        X_train, Y_train = loader._train_data[train_indices], loader._train_labels[train_indices]
        X_valid, Y_valid = loader._train_data[val_indices], loader._train_labels[val_indices]

        print(f"Training Set Size: {X_train.shape[0]}")
        print(f"Validation Set Size: {X_valid.shape[0]}")

        # Create a new network instance for each fold
        # to ensure that training starts from scratch
        mynet = Network(input_neurons, hidden_neurons, output_neurons)

        # Train the network
        mynet.fit(X_train, Y_train, X_valid, Y_valid, cross_entropy, epoch_number, eta, patience)

        # Initialize the best model with the first trained model
        best_model = mynet.copy_network() if i == 0 else best_model

        # Save the validation accuracy of this fold
        Z_valid = mynet.forward_propagation(X_valid)
        curr_acc = mynet.get_accuracy(Z_valid, Y_valid)
        validation_accuracies.append(curr_acc)

        if curr_acc > validation_accuracies[-1] and len(validation_accuracies) > 0 and i > 0:
            best_model = mynet.copy_network()  # Save the best model if current accuracy is higher

        print(f"Validation Accuracy of Fold {i+1}: {curr_acc:.15f}")

    # Calculate the final results
    mean_accuracy = np.mean(validation_accuracies)
    std_accuracy = np.std(validation_accuracies)
    print(f"\nK-Fold Cross-Validation Results: Mean Accuracy = {mean_accuracy:.15f}, Std Dev = {std_accuracy:.15f}")

    if best_model is None:
        raise RuntimeError("No valid model was trained during k-fold cross-validation.")

    return best_model


def holdout_validation(loader: DataLoader, mynet: Network, epoch_number, eta, patience, train_ratio=0.8):
    """Train the network with training and validation sets."""

    # Split the data into training and validation sets
    X_train, Y_train, X_valid, Y_valid = loader.split_data(train_ratio)

    # Print the architecture of the network
    mynet.get_info()

    # Train the network
    mynet.fit(X_train, Y_train, X_valid, Y_valid, cross_entropy, epoch_number, eta, patience)

    # Evaluate the final model on the validation set
    Z_valid = mynet.forward_propagation(X_valid)

    print(f"Validation Set accuracy: {mynet.get_accuracy(Z_valid, Y_valid):.15f}")


def perform_inference(mynet: Network, X_test, Y_test):
    """Perform inference on a single test image and visualize the result."""
    random_index = np.random.randint(0, X_test.shape[0])
    img_test = X_test[random_index]  # Get a single test image

    # Get the gold label for the test image
    true_label = np.argmax(Y_test[random_index])  # Labels are one-hot encoded

    # Infer the label of the test image
    Y_pred = mynet.forward_propagation(img_test)
    predicted_label = np.argmax(Y_pred)  # Convert one-hot to label

    print(f"\nTrue label: {true_label}")
    print(f"Predicted label: {predicted_label}")

    # Visualize the test image and its prediction
    img_test = img_test.reshape(28, 28)  # Assuming MNIST images are 28x28
    plt.imshow(img_test, cmap="gray")
    plt.title(f"True label: {true_label} | Predicted: {predicted_label}")
    plt.axis("off")  # Hide the axes
    plt.show()


def main():
    # Data loading and preprocessing
    loader = DataLoader("./MNIST")

    # Get the training and test data using the DataLoader
    X_train, Y_train, X_test, Y_test = loader.get_train_test_data()

    ### Set the number of input, hidden (can be adjusted) and output neurons ###
    output_neurons = Y_train.shape[1]
    input_neurons = X_train.shape[1]
    hidden_neurons = [50]
    ##############################################################################

    ### Set parameters for training ###
    eta = 0.1  # Learning rate
    epoch_number = 1000  # Number of epochs for training
    patience = 5  # Early stopping patience
    ##############################################################################

    # Create a network instance
    mynet = Network(input_neurons, hidden_neurons, output_neurons)

    ##############################################################################
    # Train the network using holdout validation
    holdout_validation(loader, mynet, epoch_number, eta, patience, train_ratio=0.8)
    ##############################################################################

    # Example usage of the network's forward propagation method with random test image
    # perform_inference(mynet, X_test, Y_test)
    ##############################################################################

    # Test the network on the test set
    Z = mynet.forward_propagation(X_test)  # Forward propagation on the test set
    accuracy = mynet.get_accuracy(Z, Y_test)  # Calculate accuracy on the test set
    print(f"\nTest set accuracy: {accuracy:.15f}")
    ##############################################################################

    # Train the network using k-fold cross-validation
    k = output_neurons  # Number of folds
    best_model: Network = kfold_cross_validation(loader, input_neurons, hidden_neurons, output_neurons, k, epoch_number, eta, patience)

    Z = best_model.forward_propagation(X_test)  # Forward propagation on the test set with the best model
    best_accuracy = best_model.get_accuracy(Z, Y_test)
    print(f"[K-Fold] Best model Test Set accuracy: {best_accuracy:.15f}")
    ##############################################################################


if __name__ == "__main__":
    main()
