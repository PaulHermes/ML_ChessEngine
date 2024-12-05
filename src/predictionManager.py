import threading
import time
import numpy as np
from threading import Lock, Condition
import parameters

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
        self.batch_size = parameters.prediction_batch_size
        self.queue_condition = Condition()
        self.last_enqueue = None
        self.process_interval = parameters.process_interval
        self.enforce_batch_size = parameters.enforce_batch_size

        if parameters.use_prediction_manager:
            # Start worker thread
            worker_thread = threading.Thread(target=self.worker, daemon=True)
            worker_thread.start()

    def worker(self):
        while True:
            with self.queue_condition:
                #if len(self.prediction_queue) == 0:
                    #self.queue_condition.notify_all()
                    #time.sleep(1)
                if self.last_enqueue is not None and len(self.prediction_queue) > 0:
                    elapsed_time = time.time() - self.last_enqueue
                    if (not self.enforce_batch_size) or \
                       (self.enforce_batch_size and elapsed_time >= self.process_interval):
                        self.process_prediction_queue()

    def set_neural_network(self, neural_network):
        """Set the shared neural network for predictions."""
        self.neural_network = neural_network

    def enqueue_prediction(self, node, board_input):
        """Add a node and its board input to the shared queue."""
        with self.queue_condition:
            self.prediction_queue.append((node, board_input))
            self.last_enqueue = time.time()

            #print(f"Queue size: {len(self.prediction_queue)}")
            if self.enforce_batch_size and len(self.prediction_queue) >= self.batch_size:
                print("Batch size reached. Processing queue.")
                self.process_prediction_queue()
            else:
                self.queue_condition.notify_all()

    def process_prediction_queue(self):
        """Process the prediction queue."""
        with self.queue_condition:
            if len(self.prediction_queue) == 0:
                return  # Nothing to process

            if self.enforce_batch_size:
                # Process batch of fixed size
                process_count = min(self.batch_size, len(self.prediction_queue))
            else:
                # Process all queued items
                process_count = len(self.prediction_queue)

            # Prepare batch input
            nodes, board_inputs = zip(*self.prediction_queue[:process_count])
            board_inputs = np.array(board_inputs)

            # Perform batch prediction
            #print(f"Processing batch of size {process_count}.")
            policy_outputs, value_outputs = self.neural_network.model.predict(board_inputs, verbose=0)

            # Store results for each node
            for i, node in enumerate(nodes):
                policy_output = policy_outputs[i]
                value_output = value_outputs[i][0]
                self.prediction_results[node] = (policy_output, value_output)

            # Clear processed items from the queue
            self.prediction_queue = self.prediction_queue[process_count:]

            # Notify waiting threads that predictions are ready
            self.queue_condition.notify_all()

    def get_predictions_for_node(self, node):
        """Retrieve predictions for a specific node."""
        with self.queue_condition:
            while node not in self.prediction_results:
                self.queue_condition.wait()  # Wait for a signal or timeout
            return self.prediction_results[node]

    def remove_node(self, node):
        with self.queue_condition:
            if node in self.prediction_results:
                del self.prediction_results[node]
