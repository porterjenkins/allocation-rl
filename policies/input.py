import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Dict
from stable_baselines.common.policies import mlp_extractor, nature_cnn
from collections import OrderedDict

def observation_input(ob_space, batch_size=None, name='Ob', scale=False, reuse=False):
    """
    Build observation input with encoding depending on the observation space type

    When using Box ob_space, the input will be normalized between [1, 0] on the bounds ob_space.low and ob_space.high.

    :param ob_space: (Gym Space) The observation space
    :param batch_size: (int) batch size for input
                       (default is None, so that resulting input placeholder can take tensors with any batch size)
    :param name: (str) tensorflow variable name for input placeholder
    :param scale: (bool) whether or not to scale the input
    :param reuse: (bool)
    :return: (TensorFlow Tensor, TensorFlow Tensor) input_placeholder, processed_input_tensor
    """
    if isinstance(ob_space, Discrete):
        observation_ph = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
        processed_observations = tf.cast(tf.one_hot(observation_ph, ob_space.n), tf.float32)
        return observation_ph, processed_observations

    elif isinstance(ob_space, Box):
        observation_ph = tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=ob_space.dtype, name=name)
        processed_observations = tf.cast(observation_ph, tf.float32)
        # rescale to [1, 0] if the bounds are defined
        if (scale and
           not np.any(np.isinf(ob_space.low)) and not np.any(np.isinf(ob_space.high)) and
           np.any((ob_space.high - ob_space.low) != 0)):

            # equivalent to processed_observations / 255.0 when bounds are set to [255, 0]
            processed_observations = ((processed_observations - ob_space.low) / (ob_space.high - ob_space.low))
        return observation_ph, processed_observations

    elif isinstance(ob_space, MultiBinary):
        observation_ph = tf.placeholder(shape=(batch_size, ob_space.n), dtype=tf.int32, name=name)
        processed_observations = tf.cast(observation_ph, tf.float32)
        return observation_ph, processed_observations

    elif isinstance(ob_space, MultiDiscrete):
        observation_ph = tf.placeholder(shape=(batch_size, len(ob_space.nvec)), dtype=tf.int32, name=name)
        processed_observations = tf.concat([
            tf.cast(tf.one_hot(input_split, ob_space.nvec[i]), tf.float32) for i, input_split
            in enumerate(tf.split(observation_ph, len(ob_space.nvec), axis=-1))
        ], axis=-1)
        return observation_ph, processed_observations

    elif isinstance(ob_space, Dict):
        ob_space_dict = list(OrderedDict(ob_space.spaces))
        ob_space_length = np.array([np.prod(np.array(ob_space[key].shape)) for key in ob_space_dict])

        observation_ph = tf.placeholder(shape=(batch_size, np.sum(ob_space_length)), dtype=tf.float32, name=name)

        observation_day_ph = observation_ph[:, :ob_space_length[1]]
        processed_observation_day = tf.cast(observation_day_ph, tf.float32)

        # observation_board_ph = observation_ph[:, (ob_space_length[1]+1):(ob_space_length[1]+ob_space_length[0])]
        # processed_observation_board = tf.cast(observation_board_ph, tf.float32)
        # # rescale to [1, 0] if the bounds are defined
        # if (scale and
        #         not np.any(np.isinf(ob_space["board_config"].low)) and
        #         not np.any(np.isinf(ob_space["board_config"].high)) and
        #         np.any((ob_space["board_config"].high - ob_space["board_config"].low) != 0)):
        #     # equivalent to processed_observations / 255.0 when bounds are set to [255, 0]
        #     processed_observation_board = ((processed_observation_board - ob_space["board_config"].low) /
        #                                     (ob_space["board_config"].high - ob_space["board_config"].low))

        observation_prevsales_ph = observation_ph[:, -ob_space_length[-1]:]
        processed_observation_prevsales = tf.cast(observation_prevsales_ph, tf.float32)
        # rescale to [1, 0] if the bounds are defined
        if (scale and
                not np.any(np.isinf(ob_space["prev_sales"].low)) and
                not np.any(np.isinf(ob_space["prev_sales"].high)) and
                np.any((ob_space["prev_sales"].high - ob_space["prev_sales"].low) != 0)):
            # equivalent to processed_observations / 255.0 when bounds are set to [255, 0]
            processed_observation_prevsales = ((processed_observation_prevsales - ob_space["prev_sales"].low) /
                                            (ob_space["prev_sales"].high - ob_space["prev_sales"].low))

        # TODO: these should be in params
        net_arch = None
        act_fun = tf.tanh

        if net_arch is None:
            net_arch = [32, 16]

        with tf.variable_scope("input_embedding", reuse=reuse):
            # with tf.variable_scope("board_embed", reuse=reuse):
            #     board_latent, _ = mlp_extractor(tf.layers.flatten(processed_observation_board), net_arch, act_fun)
            with tf.variable_scope("prevsales_embed", reuse=reuse):
                prevsales, _ = mlp_extractor(tf.layers.flatten(processed_observation_prevsales), net_arch, act_fun)
            processed_observations = tf.concat([processed_observation_day,
                                                # board_latent,
                                                prevsales],
                                               axis=-1, name="final_obs")
        # TODO: watch out! the processed observation is passed as observation_ph
        return observation_ph, processed_observations