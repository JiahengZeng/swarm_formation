import envfile
import tensorflow as tf
import numpy as np
import argparse
"""项目测试文件：该项目的目标是完成五种形状编队的无人机强化学习的编队重构，以应对当无人机受到损失后的队形维持问题；
五种队形包括：一字型、二字型、三角形、纵列式、双纵列式；
开发人员：Central South University-JiaHeng Zeng"""

parser = argparse.ArgumentParser(description='For formation envfile test')
parser.add_argument('--env', help='choose a env', type=str, default='uavFormation-v0')
parser.add_argument('--render', help='whether to render', type=int, default=1)
parser.add_argument('--record', help='weather to record', type=int, default=0)
parser.add_argument('--horizon', help='total steps to stop the game', type=int, default=2000)  # 总共编队飞行的时间步长
parser.add_argument('--agents_number', help='the number of UAVs', type=int, default=20)
parser.add_argument('--targets_number', help='the number of targets', type=int, default=20)
parser.add_argument('--log_dir', help='the path to save the record data', type=str, default='.')
parser.add_argument('--map', type=str, default='emptyMap')
parser.add_argument('--LR_A', type=float, default=1e-4)
parser.add_argument('--LR_C', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--load', type=bool, default=True)
parser.add_argument('--replace_iter_a', type=int, default=1100)
parser.add_argument('--replace_iter_c', type=int, default=1000)
parser.add_argument('--memory_capacity', type=int, default=5000)
args = parser.parse_args()

env = envfile.make(args.env,
                   render=args.render,
                   record=args.record,
                   mapID=args.map,
                   directory=args.log_dir,
                   horizon=args.horizon,
                   num_agents=args.agents_number,
                   num_targets=args.targets_number,
                   is_training=False)

STATE_DIM = 4
ACTION_DIM = 2
ACTION_BOUND = [-1, 1]

with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')  # None 都是为了考虑batch_size服务


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):   # 定义一个变量可共享区域
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic  也就是Actor的target网络
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
        self.replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]  # 把e的值赋给t

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run(self.replace)
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)
        self.replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run(self.replace)
        self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


sess = tf.Session()

actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], args.LR_A, args.replace_iter_a)
critic = Critic(sess, STATE_DIM, ACTION_DIM, args.LR_C, args.gamma, args.replace_iter_c, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(args.memory_capacity, dims=2 * STATE_DIM + ACTION_DIM + 1)
saver = tf.train.Saver()
path = './ddpg_tensorflow'

if args.load:
    saver.restore(sess, tf.train.latest_checkpoint(path))


def main():
    done = {'__all__': False}

    obs = env.reset(nb_agents=args.agents_number, nb_targets=args.targets_number)
    while not done['__all__']:
        if args.render:
            env.render()
        action_dictionary = {}
        for agent_id, _ in obs.items():
            action_dictionary[agent_id] = actor.choose_action(obs[agent_id][0])  # [x, x, x]  ndarray
            if agent_id == 'agent-1':
                print(action_dictionary[agent_id], 'ddddd', obs[agent_id][0])
        obs, reward, done, information = env.step(action_dictionary)
    print('the whole game has shut down!')


if __name__ == '__main__':
    main()
