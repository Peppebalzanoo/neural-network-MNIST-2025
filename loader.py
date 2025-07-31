import numpy as np


class DataLoader:
    def __init__(self, data_path: str):
        """Initialize the DataLoader with the path to the dataset."""
        self._data_path = data_path
        self._train_data = None
        self._test_data = None
        self._train_labels = None
        self._test_labels = None
        self._load_data()

    def _load_data(self):
        """Load the dataset from the specified path."""
        if not self._data_path:
            raise ValueError("Data path must be specified.")
        try:
            train_data = np.loadtxt(self._data_path + "/mnist_train.csv", delimiter=",")
            test_data = np.loadtxt(self._data_path + "/mnist_test.csv", delimiter=",")

            if train_data is None or test_data is None:
                raise ValueError("Data files not found or are empty.")
            if train_data.size == 0 or test_data.size == 0:
                raise ValueError("Loaded data is empty. Please check the dataset files.")

            # Assuming the first column contains labels and one-hot encode them
            self._train_labels = self.one_hot_encode(train_data[:, 0])
            self._test_labels = self.one_hot_encode(test_data[:, 0])

            # Normalize the data and remove the label column
            self._train_data = (train_data[:, 1:]) / 255
            self._test_data = (test_data[:, 1:]) / 255

        except Exception as e:
            raise IOError(f"Error loading data: {e}")

    def one_hot_encode(self, labels):
        num_classes = int(np.max(labels)) + 1
        one_hot_labels = np.zeros((labels.size, num_classes), dtype=int)
        idxs_of_labels = labels.astype(int)
        one_hot_labels[np.arange(labels.size), idxs_of_labels] = 1
        return one_hot_labels

    def get_train_test_data(self):
        """Return the loaded dataset: (X_train, Y_train, X_test, Y_test)."""
        if self._train_data is None or self._test_data is None:
            raise ValueError("Data not loaded. Please call _load_data() first.")
        if self._train_labels is None or self._test_labels is None:
            raise ValueError("Labels not loaded. Please check the dataset files.")
        return self._train_data, self._train_labels, self._test_data, self._test_labels

    def split_data(self, train_ratio: float = 0.8):
        """Split the training data into training set and validation set."""
        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be between 0 and 1.")

        if self._train_data is None or self._train_labels is None:
            raise ValueError("Training data or labels not loaded. Please call _load_data() first.")

        # Calculate the number of training samples based on the train_ratio
        num_train_samples = int(len(self._train_data) * train_ratio)
        # Shuffle the indices to ensure random splitting
        indices = np.random.permutation(len(self._train_data))
        train_indices = indices[:num_train_samples]
        val_indices = indices[num_train_samples:]
        # Split the data and labels into training and validation sets
        train_set = self._train_data[train_indices]
        train_labels = self._train_labels[train_indices]
        valid_set = self._train_data[val_indices]
        valid_labels = self._train_labels[val_indices]

        return train_set, train_labels, valid_set, valid_labels

    def create_stratified_k_folds(self, Y_train, k):
        """Create k stratified folds from the training set labels."""
        # Initialize k empty lists, one for each fold
        fold_indices = [[] for _ in range(k)]

        # Convert labels from one-hot to single value (e.g. from [0,0,0,1..] to 3)
        labels = np.argmax(Y_train, axis=1)
        # Get the number of classes
        num_classes = Y_train.shape[1]

        for class_id in range(0, num_classes):
            # Find all indices of samples that belong to this class
            indices_for_class = np.where(labels == class_id)[0]

            # Randomly shuffle the indices of this class
            np.random.shuffle(indices_for_class)

            # Split the shuffled indices into k "chunks" (parts) that are almost equal
            class_chunks = np.array_split(indices_for_class, k)

            # Assign each chunk to a different fold
            for i in range(k):
                # We use .extend() on a Python list, which is much more efficient
                fold_indices[i].extend(class_chunks[i])

        # Convert the index lists to NumPy arrays and shuffle them one last time
        final_folds = []
        for i in range(k):
            # Convert to NumPy array and shuffle
            fold_array = np.array(fold_indices[i], dtype=int)
            # Shuffle the fold array to ensure randomness
            np.random.shuffle(fold_array)
            # Append to the final list of folds
            final_folds.append(fold_array)

        return final_folds
