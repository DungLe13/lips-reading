import random
import numpy as np
import tensorflow as tf

class PolicyGradientActorCritic(object):
    def __init__(self, session, optimizer, actor_network, critic_network,
                 state_dim, num_actions,
                 init_exp = 0.1,           # initial exploration prob
                 final_exp = 0.0,          # final exploration prob
                 anneal_steps = 1000,      # N steps for annealing exploration
                 discount_factor = 0.99,   # discount future rewards
                 reg_param = 0.001,        # regularization constants
                 max_gradient = 5,         # max gradient norm
                 summary_writer = None,
                 summary_every = 100):
        # tensorflow machinery
        self.session = session
        self.optimizer = optimizer
        self.summary_writer = summary_writer

        # model components
        self.actor_network = actor_network
        self.critic_network = critic_network

        # training parameters
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.max_gradient = max_gradient
        self.reg_param = reg_param

        # exploration parameters
        self.exploration = init_exp
        self.init_exp = init_exp
        self.final_exp = final_exp
        self.anneal_steps = anneal_steps

        # counters
        self.train_iteration = 0

        # rollout buffer
        self.state_buffer = []
        self.reward_buffer = []
        self.action_buffer = []

        # create and initialize variables
        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.session.run(tf.variables_initializer(var_lists))

        # make sure all variables are initialized
        self.session.run(tf.assert_variables_initialized())

        if self.summary_writer is not None:
            self.summary_writer.add_graph(self.session.graph)
            self.summary_every = summary_every

    def create_variables(self):
        with tf.name_scope("model_inputs"):
            self.states = tf.placeholder(dtype=tf.float32, shape=(None, self.state_dim), name="states")

        # rollout action based on current policy
        with tf.name_scope("predict_actions"):
            # initialize actor-critic network
            with tf.variable_scope("actor_network"):
                self.policy_outputs = self.actor_network(self.states)
            with tf.variable_scope("critic_network"):
                self.value_outputs = self.critic_network(self.states)

            # predict actions from policy network
            self.action_scores = tf.identity(self.policy_outputs, name="action_scores")
            self.predicted_actions = tf.multinomial(self.action_scores, 1)

        # get variable list
        actor_network_variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_network")
        critic_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_network")

        # compute loss and gradients
        with tf.name_scope("compute_pg_gradients"):
            # gradients for selecting action from policy network
            self.taken_actions = tf.placeholder(dtype=tf.int32, shape=(None,), name="taken_actions")
            self.discounted_rewards = tf.placeholder(dtype=tf.float32, shape=(None,), name="discounted_rewards")

            with tf.variable_scope("actor_network", reuse=True):
                self.logprobs = self.actor_network(self.states)

            with tf.variable_scope("critic_network", reuse=True):
                self.estimated_values = self.critic_network(self.states)

            # compute policy loss and regularization loss
            self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logprobs, labels=self.taken_actions)
            self.pg_loss = tf.reduce_mean(self.cross_entropy_loss)
            self.actor_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in actor_network_variables])
            self.actor_loss = self.pg_loss + self.reg_param * self.actor_reg_loss

            # compute actor gradients
            self.actor_gradients = self.optimizer.compute_gradients(self.actor_loss, actor_network_variables)
            # compute advantages A(s) = R - V(s)
            self.advantages = tf.reduce_sum(self.discounted_rewards - self.estimated_values)
            # compute policy gradients
            for i, (grad, var) in enumerate(self.actor_gradients):
                if grad is not None:
                    self.actor_gradients[i] = (grad * self.advantages, var)

            # compute critic gradients
            self.mean_square_loss = tf.reduce_mean(tf.square(self.discounted_rewards - self.estimated_values))
            self.critic_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in critic_network_variables])
            self.critic_loss = self.mean_square_loss + self.reg_param * self.critic_reg_loss
            self.critic_gradients = self.optimizer.compute_gradients(self.critic_loss, critic_network_variables)

            # collect all gradients
            self.gradients = self.actor_gradients + self.critic_gradients

            # gradients clipping by norm
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)

            # summarize gradients
            for grad, var in self.gradients:
                tf.summary.histogram(var.name, var)
                if grad is not None:
                    tf.summary.histogram(var.name + '/gradients', grad)

            # emit summaries
            tf.summary.histogram("estimated_values", self.estimated_values)
            tf.summary.scalar("actor_loss", self.actor_loss)
            tf.summary.scalar("critic_loss", self.critic_loss)
            tf.summary.scalar("reg_loss", self.actor_reg_loss + self.critic_reg_loss)

        # training update
        with tf.name_scope("train_actor_critic"):
            # apply gradients to update actor network
            self.train_op = self.optimizer.apply_gradients(self.gradients)

        self.summarize = tf.summary.merge_all()
        self.no_op = tf.no_op()

    def sampleAction(self, states):
        def softmax(y):
            maxy = np.amax(y)
            e = np.exp(y - maxy)
            return e / np.sum(e)

        # epsilon-greedy exploration strategy
        if random.random() < self.exploration:
            return random.randint(0, self.num_actions-1)
        else:
            action_scores = self.session.run(self.action_scores, {self.states: states})[0]
            action_probs  = softmax(action_scores) - 1e-5
            action = np.argmax(np.random.multinomial(1, action_probs))
            return action

    def updateModel(self):
        N = len(self.reward_buffer)
        r = 0            # use discounted reward to approx. Q-value

        # compute discounted future rewards
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            r = self.reward_buffer[t] + self.discount_factor * r
            discounted_rewards[t] = r

        # whether to calculate summaries
        calculate_summaries = self.summary_writer is not None and self.train_iteration % self.summary_every == 0

        # update policy network with the rollout in batches
        for t in range(N-1):
            # prepare inputs
            states = self.state_buffer[t][np.newaxis, :]
            actions = np.array([self.action_buffer[t]])
            rewards = np.array([discounted_rewards[t]])

            # perform one update of training
            _, summary_str = self.session.run([self.train_op, self.summarize if calculate_summaries else self.no_op],
                                              {self.states: states, self.taken_actions: actions, self.discounted_rewards: rewards})

            # emit summaries
            if calculate_summaries:
                self.summary_writer.add_summary(summary_str, self.train_iteration)

        self.annealExploration()
        self.train_iteration += 1

        # clean up
        self.cleanUp()

    def annealExploration(self, strategy='linear'):
        ratio = max((self.anneal_steps - self.train_iteration) / float(self.anneal_steps), 0)
        self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp

    def resetModel(self):
        self.cleanUp()
        self.train_iteration = 0
        self.exploration = self.init_exp
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.session.run(tf.variables_initializer(var_lists))

    def storeRollout(self, state, action, reward):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)

    def cleanUp(self):
        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []
