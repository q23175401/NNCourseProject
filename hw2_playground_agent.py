from hw2_playground import Playground
from hw2_utils import resource_path
from NN.layer.activation.relu import ReLU
from ReinforcementLearning.QLearning.ddqn_agent import DDqnAgent
from ReinforcementLearning.QLearning.q_base_agent import QBaseAgent
import numpy as np
from NN.layer import DenseLayer, Flatten, Tanh
from NN.network import Network
from NN.optimizer import Adam
from NN.loss import LeastSquaredError


def build_my_qnet(n_actions, input_shape, learning_rate):
    in_units = int(np.prod(input_shape))
    layer_list = [
        Flatten(input_shape),
        DenseLayer(in_units, 64),
        Tanh(),
        DenseLayer(64, 32),
        ReLU(),

        DenseLayer(32, n_actions),
    ]

    myQNet = Network(
        layer_list=layer_list,
        optimizer=Adam(learning_rate),
        loss_func=LeastSquaredError(),
    )

    return myQNet


def build_my_agent(n_actions, input_shape):
    # hyperparameters
    LEARNING_RATE = 1e-4
    UPDATE_TARGET_NET_STEPS = 1000
    epsilon = 1
    epsilon_decay = 0.998
    epsilon_end = 0.01

    # replay buffer for training
    MAX_BUFFER_SIZE = 100000
    MIN_DATA_TO_COLLECT = 20000

    def build_my_net_func():
        return build_my_qnet(n_actions, input_shape, LEARNING_RATE)

    agent = DDqnAgent(
        n_actions=n_actions,
        input_shape=input_shape,
        learning_rate=LEARNING_RATE,
        update_target_steps=UPDATE_TARGET_NET_STEPS,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_end=epsilon_end,
        max_buffer_size=MAX_BUFFER_SIZE,
        min_data_to_collect=MIN_DATA_TO_COLLECT,
        build_q_network_func=build_my_net_func
    )
    return agent


def train_save_agent(playground=None, agent=None, save_agent_name=None):
    collect_env = playground if playground else Playground()

    # training parameters
    BATCH_SIZE = 32
    DISCOUNT_FACTOR = 0.99
    N_EPISODES = 30000
    SHOW_RESULT = 30

    agent = agent if agent else build_my_agent(
        collect_env.n_actions, np.shape(collect_env.state))

    agent.train(collect_env,
                N_EPISODES=N_EPISODES,
                batch_size=BATCH_SIZE,
                DISCOUNT_FACTOR=DISCOUNT_FACTOR,
                show_result=SHOW_RESULT)

    save_agent_name = save_agent_name if save_agent_name else resource_path(
        f'hw2_agent_saved/accumulate.pickle')
    agent.save_agent(save_agent_name)


def load_my_agent(load_agent_name=None):
    try:
        load_agent_name = load_agent_name if load_agent_name else resource_path(
            f'hw2_agent_saved/accumulate.pickle')
        agent = QBaseAgent.load_agent(load_agent_name)
    except FileNotFoundError:
        p = Playground()
        n_actions = p.n_actions
        input_shape = np.shape(p.state)
        agent = build_my_agent(n_actions=n_actions, input_shape=input_shape)
    return agent


if __name__ == "__main__":
    train_save_agent()

    # evaluate load agent
    # evaluate_env = PlayGround()
    # agent = QBaseAgent.load_agent(f'hw2_agent_saved/{20000}')
    # agent.evaluate_policy(evaluate_env, 100)
