import numpy as np
from .q_base_agent import QBaseAgent


class DDqnAgent(QBaseAgent):
    def __init__(self, n_actions, input_shape, learning_rate, update_target_steps, epsilon, epsilon_decay, epsilon_end, max_buffer_size, min_data_to_collect, build_q_network_func=None):
        super().__init__(n_actions, input_shape, learning_rate, update_target_steps, epsilon,
                         epsilon_decay, epsilon_end, max_buffer_size, min_data_to_collect, build_q_network_func)

    def train_one_batch(self, batch_size, DISCOUNT_FACTOR):
        if len(self.replay_buffer) < batch_size:
            return False  # 返回沒有訓練成功 # 至少要有一個batch的量才可以開始訓練

        one_batch = self.replay_buffer.get_batch(batch_size)
        current_states_in_one_batch = np.array(
            [traj.cur_state for traj in one_batch])
        current_q_predicted = self.value_net.Predict(
            current_states_in_one_batch)

        next_states_in_one_batch = np.array(
            [traj.next_state for traj in one_batch])
        next_q_predicted = self.value_net.Predict(next_states_in_one_batch)
        actions_taken_by_prediction = np.argmax(next_q_predicted, axis=1)

        next_q_target = self.target_net.Predict(next_states_in_one_batch)

        for ti, traj_data in enumerate(one_batch):
            _, action, reward, _, done = traj_data

            target_q = reward if done else reward + DISCOUNT_FACTOR * \
                next_q_target[ti, actions_taken_by_prediction[ti]]

            # change predicted values to target value
            current_q_predicted[ti, action] = target_q

        X = current_states_in_one_batch  # input states
        y = current_q_predicted     # need to output this answer

        self.value_net.train_on_batch(X, y)
        return True  # 返回有訓練成功
