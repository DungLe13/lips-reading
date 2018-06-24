## Double Deep Q-Network (DDQN)

### A. Definitions

Q-Learning is a model-free learning algorithm, in which the agent knows nothing about the state-transition and reward models. However, the agent will discover what are the good and bad actions by trial and error.

**The basic idea of Q-Learning is to approximate the state-action pairs Q-function from the samples of Q(s, a) that we observe during interaction with the enviornment.**

In order to transform an ordinary Q-Network into a DQN, the following improvements are made:

1. Going from a single-layer network to a multi-layer neural network. (In this case, RNN with LSTM cells is used instead of CNN.)
2. Implementing Experience Replay, which will allow our network to train itself using stored memories from it’s experience.
3. Utilizing a second “target” network, which we will use to compute target Q-values during our updates.

The main intuition behind Double DQN is that the regular DQN often overestimates the Q-values of the potential actions to take in a given state. A simple trick to correct this: instead of taking the max over Q-values when computing the target-Q value for our training step, we **use our primary network to chose an action, and our target network to generate the target Q-value for that action**.

### B. Files

- The two files `neural_q_network.py` and `sample.py` serve as an example of how DDQN is implemented for the CartPole task. Credit: [Yuke Zhu](https://github.com/yukezhu)
- The two files `lips_reading_LSTM.py` and `DDQN.py` are the modifications for the Lip Reading task using DDQN and RNN with LSTM as the policy network.

### C. Sources

- More on the definitions of Double Deep Q-Network (DDQN): [[1](https://arxiv.org/abs/1509.06461)], [[2](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df)], [[3](https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa)]
- More on the implementation of DDQN: [[4](https://github.com/yukezhu/tensorflow-reinforce)]