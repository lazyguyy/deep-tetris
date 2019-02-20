import gym
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from gym.envs.registration import register


def onehot(label, num_labels):
    result = np.zeros((1, num_labels))
    result[0, label] = 1
    return result


register(
    id='MyFrozenLake-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': True},
    max_episode_steps=100,
    reward_threshold=0.78
)

env = gym.make("MyFrozenLake-v0")
num_states, num_actions = 16, 4

dtype = tf.float64

X = tf.placeholder(shape=[1, num_states], dtype=dtype)
Q = tf.placeholder(shape=[1, num_actions], dtype=dtype)

Y = tf.layers.dense(X, num_actions)
L = tf.reduce_mean(tf.square(Y - Q))
S = tf.train.GradientDescentOptimizer(0.5).minimize(L)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_episodes = 2 ** 13
    gamma, epsilon = 0.999, 0.1

    it = tqdm(range(num_episodes))
    for episode in it:
        done, state_curr = False, env.reset()

        while not done:
            feed_dict = {X: onehot(state_curr, num_states)}
            qualities = sess.run(Y, feed_dict=feed_dict)
            action = np.argmax(qualities[0])

            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()

            state_next, reward, done, _ = env.step(action)

            feed_dict = {X: onehot(state_next, num_states)}
            qualities_next = sess.run(Y, feed_dict=feed_dict)
            qualities[0, action] = reward + gamma * np.max(qualities_next)
            feed_dict = {X: onehot(state_curr, num_states), Q: qualities}
            sess.run(S, feed_dict=feed_dict)

            state_curr = state_next

        if episode % 2 ** 8 == 0:
            it.set_description(f"{sess.run(L, feed_dict=feed_dict):0.5f}")

    num_test_games, num_won, render = 2 ** 10, 0, False
    # num_test_games, num_won, render = 1, 0, True
    for test_game in range(num_test_games):
        done, state = False, env.reset()
        while not done:
            if render:
                env.render()
            feed_dict = {X: onehot(state, num_states)}
            qualities = sess.run(Y, feed_dict=feed_dict)
            action = np.argmax(qualities[0])
            state, reward, done, _ = env.step(action)
        if render:
            env.render()
        num_won += state == num_states - 1
    print(num_won / num_test_games)
