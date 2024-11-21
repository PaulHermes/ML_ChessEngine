import numpy as np
from threading import Lock, Condition


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
        self.prediction_results = {}
        self.neural_network = None  # To be set explicitly
        self.batch_size = 512
        self.queue_condition = Condition()

    def set_neural_network(self, neural_network):
        """Set the shared neural network for predictions."""
        self.neural_network = neural_network

    def enqueue_prediction(self, node, board_input):
        """Add a node and its board input to the shared queue."""
        with self.queue_condition:
            self.prediction_queue.append((node, board_input))
            print(f"Queue size: {len(self.prediction_queue)}")

            # Trigger batch processing if the batch size is reached
            if len(self.prediction_queue) >= self.batch_size:
                print("Batch size reached. Processing queue.")
                self.process_prediction_queue()
            else:
                self.queue_condition.notify_all()  # Notify waiting threads

    def process_prediction_queue(self):
        """Process the prediction queue in batches."""
        with self.queue_condition:
            if len(self.prediction_queue) == 0:
                return  # Nothing to process

            # Determine the batch size for this run
            process_count = min(self.batch_size, len(self.prediction_queue))

            # Prepare batch input
            nodes, board_inputs = zip(*self.prediction_queue[:process_count])
            board_inputs = np.array(board_inputs)

            # Perform batch prediction
            print(f"Processing batch of size {process_count}.")
            policy_outputs, value_outputs = self.neural_network.model.predict(board_inputs, verbose=1)

            # Store results for each node
            for i, node in enumerate(nodes):
                policy_output = policy_outputs[i]
                value_output = value_outputs[i][0]
                self.prediction_results[node] = (policy_output, value_output)

            # Clear processed items from the queue
            self.prediction_queue = self.prediction_queue[process_count:]

            # Notify waiting threads that predictions are ready
            self.queue_condition.notify_all()

    def get_predictions_for_node(self, node, timeout=5):
        """Retrieve predictions for a specific node."""
        with self.queue_condition:
            while node not in self.prediction_results:
                self.queue_condition.wait(timeout)  # Wait for a signal or timeout

                # Check if predictions are still not available
                if node not in self.prediction_results:
                    raise TimeoutError(f"Prediction for node {node} not available within {timeout} seconds.")

            return self.prediction_results.pop(node, None)
