from network import Network
from loader import DataLoader
from error import cross_entropy
import numpy as np


def run_kfold_cross_validation(loader: DataLoader, input_neurons, output_neurons, k, epoch_number, eta):
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
        mynet.fit(X_train, Y_train, X_valid, Y_valid, cross_entropy, epoch_number, eta)

        # Save the validation accuracy of this fold
        Z_valid = mynet.forward_propagation(X_valid)
        acc = mynet.get_accuracy(Z_valid, Y_valid)
        validation_accuracies.append(acc)
        print(f"Validation Accuracy of Fold {i+1}: {acc:.4f}")

    # 3. Calculate and print the final results
    mean_accuracy = np.mean(validation_accuracies)
    std_accuracy = np.std(validation_accuracies)

    return mean_accuracy, std_accuracy


def main():
    # --- Data loading (as before) ---
    loader = DataLoader("./MNIST")

    # We use the entire training set for cross-validation
    X_train, Y_train, _, _ = loader.get_train_test_data()

    output_neurons = Y_train.shape[1]
    input_neurons = X_train.shape[1]

    epoch_number = 50  # Number of epochs for training
    eta = 0.5  # Learning rate

    # *** Neural network training ***
    X_train, Y_train, X_valid, Y_valid = loader.split_train_data(train_ratio=0.8)

    # --- Neural network creation ---
    mynet = Network(input_neurons, [50], output_neurons)
    mynet.get_info()

    # --- Neural network training ---
    mynet.fit(X_train, Y_train, X_valid, Y_valid, cross_entropy, epoch_number, eta)

    # --- Final evaluation ---
    Z_valid = mynet.forward_propagation(X_valid)
    acc = mynet.get_accuracy(Z_valid, Y_valid)
    print(f"Final validation accuracy: {acc:.4f}")
    print("*** End Neural Network Training ***")

    # *** Cross-validation training ***
    mean_accuracy, std_accuracy = run_kfold_cross_validation(loader, input_neurons, output_neurons, output_neurons, epoch_number, eta=eta)
    print("*** End Cross-validation Training ***")

    # Print final results
    print(f"Final validation accuracy: {acc:.4f}")
    print(f"Mean Cross-validation accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")


if __name__ == "__main__":
    main()
