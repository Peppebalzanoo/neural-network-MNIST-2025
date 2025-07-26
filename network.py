import numpy as np


class Network:
    def __init__(self, input_n: int, hiddens_n: list, output_n: int, sigma: float = 0.001):
        self._input_neurons: int = input_n
        self._hidden_neurons: list = hiddens_n
        self._output_neurons: int = output_n
        self._sigma: float = sigma
        self._baiases: list = list()
        self._weights: list = list()
        self._init_params()

    def _init_params(self) -> None:
        prev_conn_num: int = self._input_neurons
        for curr_hidden in self._hidden_neurons:
            # Baiases array for each curr_hidden neurons (one bias for each neuron)
            self._baiases.append(self._sigma * np.random.normal(size=[curr_hidden, 1]))
            # Weights matrix for each curr_hidden neurons
            self._weights.append(self._sigma * np.random.normal(size=[curr_hidden, prev_conn_num]))
            prev_conn_num = curr_hidden
        # Baiases array for each output neurons (one bias for each neuron)
        self._baiases.append(self._sigma * np.random.normal(size=[self._output_neurons, 1]))
        # Weights matrix for each output neurons
        self._weights.append(self._sigma * np.random.normal(size=[prev_conn_num, self._output_neurons]))

    def _get_info(self) -> None:
        print(f"Network [{self._input_neurons} x {self._hidden_neurons} x {self._output_neurons}]")
        print(f"baiases: {self._baiases}")
        print(f"weights: {self._weights}")

    def set_decay(self) -> None:
        pass

    def fit(self) -> None:
        pass
