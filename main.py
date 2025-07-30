import numpy as np
from network import Network
from loader import DataLoader
from error import cross_entropy
import matplotlib.pyplot as plt


def kfold_cross_validation(loader: DataLoader, input_neurons, output_neurons, k, epoch_number, eta, patience):
    """Run k-fold cross-validation on the dataset."""

    validation_accuracies = []

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
        mynet = Network(input_neurons, [50], output_neurons)
        mynet.get_info()

        # Train the network
        mynet.fit(X_train, Y_train, X_valid, Y_valid, cross_entropy, epoch_number, eta, patience)

        # Save the validation accuracy of this fold
        Z_valid = mynet.forward_propagation(X_valid)
        acc = mynet.get_accuracy(Z_valid, Y_valid)
        validation_accuracies.append(acc)
        print(f"Validation Accuracy of Fold {i+1}: {acc:.4f}")

    # 3. Calculate and print the final results
    mean_accuracy = np.mean(validation_accuracies)
    std_accuracy = np.std(validation_accuracies)

    return mean_accuracy, std_accuracy


def default_validation_split(loader: DataLoader, mynet: Network, epoch_number, eta, patience, train_ratio=0.8):
    """Split the training data into training and validation sets."""
    X_train, Y_train, X_valid, Y_valid = loader.split_train_data(train_ratio)

    # Create a new network instance
    mynet.get_info()

    # Train the network
    mynet.fit(X_train, Y_train, X_valid, Y_valid, cross_entropy, epoch_number, eta, patience)

    # Evaluate the final model on the validation set
    Z_valid = mynet.forward_propagation(X_valid)
    acc = mynet.get_accuracy(Z_valid, Y_valid)
    print(f"Final validation accuracy: {acc:.4f}")


def main():
    # Data loading and preprocessing
    loader = DataLoader("./MNIST")

    # Get the training and test data using the DataLoader
    X_train, Y_train, X_test, Y_test = loader.get_train_test_data()

    # Set the number of input, hidden (can be adjusted) and output neurons
    output_neurons = Y_train.shape[1]
    input_neurons = X_train.shape[1]
    hidden_neurons = [50]

    # Set parameters for training
    eta = 0.1  # Learning rate
    epoch_number = 1000  # Number of epochs for training
    patience = 5  # Early stopping patience

    print("*** Start Neural Network Training ***")
    mynet = Network(input_neurons, hidden_neurons, output_neurons)
    default_validation_split(loader, mynet, epoch_number, eta, patience, train_ratio=0.8)
    print("*** End Neural Network Training ***")

    # Example usage of the network's forward propagation method
    total = X_test.shape[0]
    correct = 0
    for img_index in range(0, total):
        img_test = X_test[img_index]  # Get a single test image
        img_test_label = Y_test[img_index]  # Labels are one-hot encoded
        true_img_label = np.argmax(img_test_label) % output_neurons  # Convert one-hot to label

        # Infer the label of the test image
        Y_pred = mynet.forward_propagation(img_test)
        predicted_label = np.argmax(Y_pred, axis=1) % output_neurons  # Convert one-hot to label

        # Print the results
        # print(f"Test image {img_index}, converted label: {true_img_label}")
        # print(f"Predicted label: {predicted_label}")
        # print(f"Predicted probabilities: {Y_pred}")

        # Visualize the test image and its prediction
        # img_test_print = img_test.reshape(28, 28)  # Assuming MNIST images are 28x28
        # plt.imshow(img_test_print, cmap="gray")
        # plt.title(f"Etichetta Vera: {true_img_label} | Predizione: {predicted_label}")
        # plt.axis("off")  # Nasconde gli assi
        # plt.show()

        correct += 1 if true_img_label == predicted_label else 0

    print(f"Test set accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    main()
