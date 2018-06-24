## Asynchronous Actor-Critic Agents (A3C)

### A. Definitions

Actor-Critic method is one of the policy-iteration methods introduced by Google DeepMind in 2017. It was faster, simpler, more robust, and able to achieve much better scores on the standard battery of Deep RL tasks. The three main parts are broken down below:

1. **Asynchronous**: In A3C there is a global network, and multiple worker agents which each have their own set of network parameters. Each of these agents interacts with it’s own copy of the environment at the same time as the other agents are interacting with their environments.
2. **Actor-Critic**: In the case of A3C, the network will estimate both a value function V(s) (how good a certain state is to be in) and a policy π(s) (a set of action probability outputs).
3. **Advantage**: The insight of using advantage estimates rather than just discounted returns is to *allow the agent to determine not just how good its actions were, but how much better they turned out to be than expected*. Intuitively, this allows the algorithm to focus on where the network’s predictions were lacking.

### B. Files

- The two files `actor_critic.py` and `sample.py` serve as an example of how Actor-Critic method is implemented for the Acrobot task. Credit: [Yuke Zhu](https://github.com/yukezhu)
- The two files `lips_reading_LSTM.py` and `A3C.py` are the modifications for the Lip Reading task using Actor-Critic method and RNN with LSTM as the actor and critic networks.

### C. Sources

- More on the definitions of Asynchronous Actor-Critic Agents (A3C): [[1](https://arxiv.org/abs/1602.01783)], [[2](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)]
- More on the implementation of A3C: [[3](https://github.com/yukezhu/tensorflow-reinforce)]