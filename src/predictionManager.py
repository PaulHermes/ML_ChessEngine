import numpy as np
from threading import Lock

class PredictionManager:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(PredictionManager, cls).__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self.prediction_queue = []  # Shared queue
        self.prediction_results = {}  # Results mapping
        self.neural_network = None  # To be set explicitly
        self.batch_size = 64
        self.lock = Lock()  # Lock for thread-safe operations

    def set_neural_network(self, neural_network):
        self.neural_network = neural_network

    def enqueue_prediction(self, node, board_input):
        with self.lock:
            self.prediction_queue.append((node, board_input))

            # Trigger batch prediction if queue size reaches the batch size
            if len(self.prediction_queue) >= self.batch_size:
                self.process_prediction_queue()

    def process_prediction_queue(self):
        with self.lock:
            if not self.prediction_queue:
                return

            # Prepare batch input
            nodes, board_inputs = zip(*self.prediction_queue)
            board_inputs = np.array(board_inputs)

            print(board_inputs.shape)
            # Perform batch prediction
            policy_outputs, value_outputs = self.neural_network.model.predict(board_inputs, verbose=0)

            # Store results for each node
            for i, node in enumerate(nodes):
                policy_output = policy_outputs[i]
                value_output = value_outputs[i][0]
                self.prediction_results[node] = (policy_output, value_output)

            # Clear the queue
            self.prediction_queue = []

    def get_predictions_for_node(self, node):
        with self.lock:
            return self.prediction_results.pop(node, None)
