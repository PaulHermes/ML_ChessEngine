import json
import numpy as np
import tensorflow as tf
from reinforcementLearningModel import ReinforcementLearningModel
import parameters
import os
import datetime
import chess
import util
from learning_rate_scheduler import LearningRateScheduler

class TrainingPipeline:
    def __init__(self, model, data_folder="../self_play_records", batch_size=parameters.batch_size, epochs=parameters.epochs):
        self.model = model
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_folder = "../checkpoints"

        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

    def load_data(self):
        data = []
        for file_name in os.listdir(self.data_folder):
            if file_name.endswith(".json"):
                with open(os.path.join(self.data_folder, file_name), 'r') as f:
                    game_data = json.load(f)
                    data.extend(game_data)

        states, policy_targets, value_targets = [], [], []
        for entry in data:
            state = np.array(entry["state"], dtype=np.bool_)
            states.append(state)

            move_index = self.move_to_index(entry["move"])
            policy_target = np.zeros(parameters.neural_network_output)
            policy_target[move_index] = 1
            policy_targets.append(policy_target)

            value_targets.append(entry["outcome"])

        return np.array(states), np.array(policy_targets), np.array(value_targets)

    def move_to_index(self, move_uci):
        from_square = chess.parse_square(move_uci[:2])
        to_square = chess.parse_square(move_uci[2:4])
        return from_square * 64 + to_square

    def train(self, cycle):
        util.load_latest_weights(self.model, self.checkpoint_folder)

        states, policy_targets, value_targets = self.load_data()

        lr_scheduler = LearningRateScheduler(stage="main")

        self.model.model.fit(
            x=states,
            y={'policy_head': policy_targets, 'value_head': value_targets},
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            callbacks=[lr_scheduler]
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_path = f"{self.checkpoint_folder}/model_weights_cycle_{cycle}_{timestamp}.weights.h5"

        self.model.model.save_weights(checkpoint_path)
        print(f"Model training completed and weights saved to {checkpoint_path}")


if __name__ == '__main__':
    model = ReinforcementLearningModel(parameters.neural_network_input, parameters.neural_network_output)
    model.build()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    pipeline = TrainingPipeline(model)
    cycle_number = 3
    pipeline.train(cycle=cycle_number)
