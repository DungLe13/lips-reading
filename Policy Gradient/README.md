## REINFORCE algorithm

### A. Definitions

REINFORCE is a Policy Gradient method used in Reinforcement Learning. It iteratively updates agent's parameters by computing policy gradient.

![REINFORCE algorithm](https://qph.fs.quoracdn.net/main-qimg-09daf3885ee2dac4acf4779f46b2421e.webp)

One disadvantage of REINFORCE algorithm is high variance, which can be mitigated using a baseline method.

### B. Files

- The two files `pq_reinforce.py` and `sample.py` serve as an example of how REINFORCE algorithm is implemented for the CartPole task. Credit: [Yuke Zhu](https://github.com/yukezhu)
- The two files `lips_reading_LSTM.py` and `REINFORCE.py` are the modifications for the Lip Reading task using REINFORCE algorithm and RNN with LSTM as the policy network.

### C. Sources

- More on the definitions of REINFORCE algorithm: [Sutton and Barto, Reinforcement Learning: An Introduction], [[1](https://arxiv.org/abs/1511.06732)], [[2](https://link.springer.com/article/10.1007/BF00992696)], [[3](https://arxiv.org/abs/1505.00521)]
- More on the implementation of REINFORCE: [[4](https://github.com/yukezhu/tensorflow-reinforce)]