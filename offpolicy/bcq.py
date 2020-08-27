import numpy as np
from tqdm import tqdm
import pickle

from stable_baselines.deepq.replay_buffer import ReplayBuffer
import config.config as cfg
from policies.deepq.policies import MlpPolicy
from envs.allocation_env import AllocationEnv
from envs.prior import Prior
from policies.deepq.dqn import DQN
from envs.state import State



import tensorflow as tf
import utils
import collections


# modified from BCQ in PyTorch code: https://github.com/sfujim/BCQ


def update_target_variables(target_variables,
                            source_variables,
                            tau=1.0,
                            use_locking=False,
                            name="update_target_variables"):
  """Returns an op to update a list of target variables from source variables.
  The update rule is:
  `target_variable = (1 - tau) * target_variable + tau * source_variable`.
  Args:
    target_variables: a list of the variables to be updated.
    source_variables: a list of the variables used for the update.
    tau: weight used to gate the update. The permitted range is 0 < tau <= 1,
      with small tau representing an incremental update, and tau == 1
      representing a full update (that is, a straight copy).
    use_locking: use `tf.Variable.assign`'s locking option when assigning
      source variable values to target variables.
    name: sets the `name_scope` for this op.
  Raises:
    TypeError: when tau is not a Python float
    ValueError: when tau is out of range, or the source and target variables
      have different numbers or shapes.
  Returns:
    An op that executes all the variable updates.
  """
  if not isinstance(tau, float) and not tf.is_tensor(tau):
    raise TypeError("Tau has wrong type (should be float) {}".format(tau))
  if not tf.is_tensor(tau) and not 0.0 < tau <= 1.0:
    raise ValueError("Invalid parameter tau {}".format(tau))
  if len(target_variables) != len(source_variables):
    raise ValueError("Number of target variables {} is not the same as "
                     "number of source variables {}".format(
                         len(target_variables), len(source_variables)))

  same_shape = all(trg.get_shape() == src.get_shape()
                   for trg, src in zip(target_variables, source_variables))
  if not same_shape:
    raise ValueError("Target variables don't have the same shape as source "
                     "variables.")

  def update_op(target_variable, source_variable, tau):
    if tau == 1.0:
      return target_variable.assign(source_variable, use_locking)
    else:
      return target_variable.assign(
          tau * source_variable + (1.0 - tau) * target_variable, use_locking)

  with tf.name_scope(name, values=target_variables + source_variables):
    update_ops = [update_op(target_var, source_var, tau)
                  for target_var, source_var
                  in zip(target_variables, source_variables)]
    return tf.group(name="update_all_variables", *update_ops)

class BCQNetwork:
    def __init__(self, name, state_dim, action_dim, actor_hs_list, critic_hs_list, critic_lr):
        self.name = name
        with tf.variable_scope(self.name):
            # placeholders for actor and critic networks
            self.state_ = tf.placeholder(tf.float32, [None, state_dim], name='state')
            # placeholders for actor network
            self.action_ = tf.placeholder(dtype=tf.int32, shape=[None, ], name='action')
            # placeholders for target critic soft q network
            self.reward_ = tf.placeholder(tf.float32, [None], name='rewards')
            self.discount_ = tf.placeholder(tf.float32, [None,], name='discounts')
            self.flipped_done_ = tf.placeholder(tf.float32, [None], name='flipped_dones')
            # placeholders for training generator network
            self.next_target_q_ = tf.placeholder(tf.float32, [None, action_dim], name='next_target_q')

            self.next_q_ = tf.placeholder(tf.float32, [None, action_dim], name='next_q')
            self.next_fc3_actor_= tf.placeholder(tf.float32, [None, action_dim], name='next_imt')
            self.next_action_= tf.placeholder(tf.float32, [None, action_dim], name='next_action')

            # actor network
            # input is concat of state and action (in DDPG input is just the state)
            self.fc1_actor_ = tf.contrib.layers.fully_connected(tf.concat(self.state_, axis=1),
                                                                actor_hs_list[0], activation_fn=tf.nn.relu)  # 64
            self.fc2_actor_ = tf.contrib.layers.fully_connected(self.fc1_actor_, actor_hs_list[1],
                                                                activation_fn=tf.nn.relu)
            self.fc3_actor_ = tf.contrib.layers.fully_connected(self.fc2_actor_, action_dim,
                                                                activation_fn=None)
            self.action_pred_ = tf.nn.softmax(self.fc3_actor_,axis=1)

            # value network
            self.fc1_q_ = tf.contrib.layers.fully_connected(tf.concat(self.state_, axis=1),
                                                                critic_hs_list[0], activation_fn=tf.nn.relu)  # 64
            self.fc2_q_ = tf.contrib.layers.fully_connected(self.fc1_q_, critic_hs_list[1],
                                                                activation_fn=tf.nn.relu)
            self.fc3_q_ = tf.contrib.layers.fully_connected(self.fc2_q_, action_dim,
                                                                activation_fn=tf.nn.relu)

            self.q_1_out_ = tf.contrib.layers.fully_connected(self.fc3_q_, action_dim, activation_fn=tf.nn.softmax)

            imt = tf.math.exp(self.next_action_)
            imt = tf.cast(imt/tf.math.reduce_max(imt, axis=1, keepdims=True) > 0.3, tf.float32)
            next_action = tf.reshape(tf.cast(tf.argmax(imt * self.next_q_ + (1 - imt) * (-1e8), axis=1), tf.int32), (-1, 1))

            next_action = tf.stack([tf.range(tf.shape(next_action)[0]), next_action[:, 0]], axis=-1)
            self.target_q_ = self.reward_ + \
                             self.flipped_done_ * self.discount_ * tf.gather_nd(self.next_target_q_, next_action)
            # next_state estimate is 0 if done

            # train critic with combined losses of q network and action loss
            self.total_loss_ = tf.losses.huber_loss(self.target_q_,
                                                    tf.reduce_sum(tf.multiply(self.q_1_out_,
                                                                              tf.one_hot(self.action_, depth=action_dim)
                                                                              ), axis=1)) \
                                + tf.nn.softmax_cross_entropy_with_logits(logits=self.fc3_actor_,
                                                                          labels=tf.one_hot(self.action_, depth=action_dim)) \
                                + 1e-2 * tf.reduce_sum(tf.math.pow(self.next_fc3_actor_,
                                                                   tf.fill(tf.shape(self.next_fc3_actor_), 2.0)), axis=1)
            self.total_optim_ = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(self.total_loss_)

    # get variables for actor and critic networks for target network updating
    def get_network_variables(self):
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]


class BCQ(object):

    def __init__(self, state_dim, action_dim, sess, tau=0.001, actor_hs=[64, 64], actor_lr=0.001,
                     critic_hs=[64, 64], critic_lr=0.001, dqda_clipping=None, clip_norm=False, vae_lr=0.001):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = action_dim * 2
        self.eval_eps = 0.001

        self.bcq_train = BCQNetwork("train_bcq", state_dim=state_dim, action_dim=action_dim,
                                    actor_hs_list=actor_hs, critic_hs_list=critic_hs, critic_lr=critic_lr)
        self.bcq_target = BCQNetwork("target_bcq", state_dim=state_dim, action_dim=action_dim,
                                     actor_hs_list=actor_hs, critic_hs_list=critic_hs, critic_lr=critic_lr)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        # target network update operations
        self.target_network_update_op = update_target_variables(self.bcq_target.get_network_variables(),
                                                                self.bcq_train.get_network_variables(), tau=tau)
        # intialize networks to start with the same variables:
        self.target_same_init = update_target_variables(self.bcq_target.get_network_variables(),
                                                             self.bcq_train.get_network_variables(), tau=1.0)
        self.sess.run(self.target_same_init)

    def select_action(self, state, mask=None):

        if np.random.uniform(0,1) > self.eval_eps:
            q_, action_, fc3_actor_ = self.sess.run(
                [self.bcq_train.q_1_out_, self.bcq_train.action_pred_, self.bcq_train.fc3_actor_],
                feed_dict={
                    self.bcq_train.state_: state
                }
            )
            imt = np.array(action_/np.max(action_, axis=1, keepdims=True) > 0.3)

            if mask is not None:
                imt = np.array(imt & np.array(mask, dtype=bool),dtype=int)
            return np.argmax(imt * q_ + (1 - imt) * (-1e8), axis=1)
        else:
            return np.random.randint(self.action_dim, size=state.shape[0])

    def predict(self, observation, state=None, mask=None, deterministic=True):
        if isinstance(observation, dict):
            observation = State.get_vec_observation(observation)[None]

        with self.sess.as_default():
            actions = self.select_action(observation, mask=mask)

        return actions[0], None

    def save(self, filename, directory):
        self.saver.save(self.sess, "{}/{}.ckpt".format(directory, filename))

    def load(self, filename, directory):
        self.saver.restore(self.sess, "{}/{}.ckpt".format(directory, filename))

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99):
        discount_batch = np.array([discount] * batch_size)
        stats_loss = {}
        stats_loss["actor_loss"] = 0.0
        stats_loss["critic_loss"] = 0.0
        stats_loss["vae_loss"] = 0.0
        for it in range(iterations):

            # Sample batches: done_batch has bool flipped in RL loop
            state_batch, action_batch, reward_batch, next_state_batch, flipped_done_batch = replay_buffer.sample(
                batch_size)  # already flipped done bools

            # todo add action mask

            # get next-state q, imt and i
            next_q_, next_action_, next_fc3_actor_ = self.sess.run(
                [self.bcq_train.q_1_out_, self.bcq_train.action_pred_, self.bcq_train.fc3_actor_],
                feed_dict={
                    self.bcq_train.state_: next_state_batch
                }
            )

            # get next state target q
            next_target_q_ = self.sess.run(self.bcq_target.q_1_out_, feed_dict={
                self.bcq_target.state_: next_state_batch
            })

            # train bcq network
            critic_loss, _ = self.sess.run([self.bcq_train.total_loss_, self.bcq_train.total_optim_], feed_dict={
                self.bcq_train.state_: state_batch,
                self.bcq_train.action_: action_batch,
                self.bcq_train.reward_: reward_batch,
                self.bcq_train.flipped_done_: flipped_done_batch.astype(float),
                self.bcq_train.discount_: discount_batch,
                self.bcq_train.next_target_q_: next_target_q_,
                self.bcq_train.next_q_: next_q_,
                self.bcq_train.next_fc3_actor_: next_fc3_actor_,
                self.bcq_train.next_action_: next_action_

            })

            # Update Target Networks
            self.sess.run(self.target_network_update_op)
            # get loss for stats for inner iteration
            stats_loss["critic_loss"] += critic_loss

        # return stats_loss
        stats_loss["critic_loss"] /= iterations
        return stats_loss


    #
    # def __init__(self, policy, env_model, rollout_batch_size, buffer_path, epochs, rollout, n_actions, lmbda, buffer_size=50000):
    #     self.epochs = epochs
    #     self.env_model = env_model
    #     self.rollout_batch_size = rollout_batch_size
    #     self.buffer_path = buffer_path
    #     self.rollout = rollout
    #     self.policy = policy
    #     self.n_actions = n_actions
    #     self.lmbda = lmbda
    #     self.buffer_env, self.buffer_model = self.init_buffer(self.buffer_path, buffer_size)
    #     self.n_regions = env_model.n_regions
    #     self.n_products = env_model.n_products
    #     self.env_model.reset()
    #
    # def init_buffer(self, fpath=None, buffer_size=None):
    #
    #     with open(fpath, 'rb') as f:
    #         buffer_env = pickle.load(f)
    #
    #     buffer_model = ReplayBuffer(buffer_size)
    #
    #     print("Copying environment buffer: ")
    #     for i in tqdm(range(len(buffer_env))):
    #         obs_t, action, reward, obs_tp1, done = buffer_env._storage[i]
    #         buffer_model.add(obs_t, action, reward, obs_tp1, done)
    #
    #     return buffer_env, buffer_model
    #
    #
    # def save_buffer(self, fpath="../data/mopo-buffer.p"):
    #
    #     with open(fpath, 'wb') as f:
    #         pickle.dump(self.buffer, f)
    #
    # def learn(self):
    #
    #     for i in range(self.epochs):
    #         print(f"Epoch {i}")
    #         pbar = tqdm(range(self.rollout_batch_size))
    #         for b in pbar:
    #             state = self.buffer_env.sample(batch_size=1)[0][0]
    #             #state = env.reset()
    #
    #
    #             for h in range(self.rollout):
    #                 pbar.set_description(f"batch: {b} rollout: {h}")
    #                 board_cfg = State.get_board_config_from_vec(state,
    #                                                             n_regions=self.n_regions,
    #                                                             n_products=self.n_products
    #                                                             )
    #
    #                 feasible_actions = AllocationEnv.get_feasible_actions(board_cfg)
    #                 #feasible_actions = AllocationEnv.get_feasible_actions(state["board_config"])
    #                 action_mask = AllocationEnv.get_action_mask(feasible_actions, self.n_actions)
    #
    #                 # sample action a_j ~ pi(s_j)
    #                 action, _states = self.policy.predict(state.reshape(1, -1), mask=action_mask)
    #
    #                 # compute dynamics from env model
    #                 new_state, r_hat, dones, info = self.env_model.step(action)
    #                 new_state = State.get_vec_observation(new_state)
    #
    #                 reward = self.get_penalized_reward(r_hat, self.lmbda)
    #
    #
    #                 # add (s, a, r, s') to buffer
    #                 self.buffer_model.add(obs_t=state,
    #                                       action=action,
    #                                       reward=reward,
    #                                       obs_tp1=new_state,
    #                                       done=float(dones))
    #
    #
    #
    #         # update policy with samples from D_env and D_model
    #         self.policy.update_weights(self.buffer_model)
    #     self.save_buffer()

    #
    # def get_penalized_reward(self, r, lmbda):
    #     variance = np.var(r)
    #     mean = r.mean()
    #     return mean - lmbda*variance



if __name__ == "__main__":
    prior = Prior(config=cfg.vals)
    env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True, full_posterior=True)
    policy = DQN(MlpPolicy, env, batch_size=32)

    bcq = BCQ(policy=policy,
                env_model=env,
                rollout_batch_size=10,
                epochs=100,
                rollout=10,
                n_actions = env.n_actions,
                lmbda=1e-3,
                buffer_path="../data/random-buffer.p"
                #buffer_path=None

    )

    bcq.learn()