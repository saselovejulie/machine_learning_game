import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
from collections import deque
from pacman_game import pacman
from pacman_game import pacman_utils
import cv2
import random

IMAGE_SIZE = 101  # 图片的尺寸
IMAGE_CHANNEL = 3  # 深度(色域)

INPUT_NODES = 30603  # 输出的元素的大小, IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNEL 像素点数
OUTPUT_NODES = 4  # 4分类问题, 对应PacManActions的4个枚举

# 第一层卷积层的尺寸
CONV1_DEEP = 32
CONV1_SIZE = 8

# 第二层卷积层的尺寸
CONV2_DEEP = 64
CONV2_SIZE = 5

# 第三层卷积层的尺寸
CONV3_DEEP = 64
CONV3_SIZE = 3

# 全连接层
FC_SIZE = 512

REPLAY_MEMORY = 50000  # 保存在队列中作为训练素材的长度
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
OBSERVE = 10000.  # timeSteps to observe before training
BATCH = 32
INITIAL_EPSILON = 0.1  # random moving


def create_network(input_tensor, train, regularizer):

    # 第一层卷积层, 进入图片尺寸101*101*4, 输出 101*101*32
    with tf.variable_scope("layer1-conv1"):
        conv1_weight = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNEL, CONV1_DEEP], tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("biases", [CONV1_DEEP], tf.float32, initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weight, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第二层池化层, 101*101*32 输出 51*51*32
    with tf.variable_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 声明第三层的卷积层, 进入 51*51*32, 输出 51*51*64
    with tf.variable_scope("layer3-conv2"):
        conv2_weight = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biases", [CONV2_DEEP], tf.float32, initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层池化层, 进入 51*51*64 输出 26*26*64
    with tf.variable_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 第四层输入26*26*64输出26*26*64
    with tf.variable_scope("layer5-conv3"):
        conv3_weight = tf.get_variable("weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP], tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("biases", [CONV3_DEEP], tf.float32, initializer=tf.constant_initializer(0.0))

        conv3 = tf.nn.conv2d(pool2, conv3_weight, [1, 1, 1, 1], padding="SAME")
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    # 输入26*26*64 输出 13*13*64
    with tf.variable_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 准备进入全连接层, 13*13*64=10816
    # pool_shape = [None, 10816]
    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    #
    # # 其实就是转换为[图片数量, 10816]*猜测
    # reshaped = tf.reshape(pool3, [pool_shape[0], nodes])
    reshaped = tf.reshape(pool3, [-1, nodes])

    with tf.variable_scope("layer7-fc1"):
        fc1_weight = tf.get_variable("weight", [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 全连接层增加正则化, 防止过拟合
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc1_weight))

        fc1_biases = tf.get_variable("biases", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight) + fc1_biases)

        # 只有在训练时加入dropout
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope("layer8-fc2"):
        fc2_weight = tf.get_variable("weight", [FC_SIZE, OUTPUT_NODES],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable("biases", [OUTPUT_NODES], initializer=tf.constant_initializer(0.1))

        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc2_weight))

        logit = tf.matmul(fc1, fc2_weight) + fc2_biases

    # writer = tf.train.Su

    return logit


def train_network():

    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL], name="x-input")
    train_queue = deque(maxlen=50000)

    # 正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    network = create_network(x, False, regularizer)

    # 交叉熵
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=network)
    # cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #
    # # 总损失 = 交叉熵损失 + 正则化的权重
    # loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    #
    # train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

    a = tf.placeholder("float", [None, OUTPUT_NODES])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(network, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game = pacman.PacMan()
    game.start_game()

    result, image_data, reward = game.next_step(pacman_utils.PacManActions.NOTHING)
    # 修改图片尺寸, 并调整色值域
    image_data = cv2.cvtColor(cv2.resize(image_data, (101, 101)), cv2.COLOR_BGR2HLS)
    # 像素高于阈值时，给像素赋予新值，否则，赋予另外一种颜色
    ret, image_data = cv2.threshold(image_data, 1, 255, cv2.THRESH_BINARY)
    s_t = np.reshape(image_data, (101, 101, 3))
    # image_data = np.append(image_data, image_data[:, :, :2], axis=2)
    # s_t = np.stack((image_data, image_data, image_data, image_data), axis=2)
    # s_t = np.append(image_data, image_data[:, :, :1], axis=2)

    sess = tf.InteractiveSession()

    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    epsilon = INITIAL_EPSILON
    step = 0
    while True:
        network_t = network.eval(feed_dict={x: [s_t]})[0]
        action = np.zeros([OUTPUT_NODES])
        if random.random() <= epsilon and step <= OBSERVE:
            print("----------Random Action----------")
            action[random.randrange(OUTPUT_NODES)] = 1
        else:
            action[np.argmax(network_t)] = 1

        state, image_data, reward = game.next_step(pacman_utils.get_action_from_array(action))
        # 修改图片尺寸, 并调整色值域
        image_data = cv2.cvtColor(cv2.resize(image_data, (101, 101)), cv2.COLOR_BGR2HLS)
        # 像素高于阈值时，给像素赋予新值，否则，赋予另外一种颜色
        ret, image_data = cv2.threshold(image_data, 1, 255, cv2.THRESH_BINARY)
        image_data = np.reshape(image_data, (101, 101, 3))
        # s_t1 = np.append(image_data, s_t[:, :, :1], axis=2)

        train_queue.append((s_t, action, reward, image_data, state))

        if step > OBSERVE:
            minibatch = random.sample(train_queue, BATCH)
            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            action_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            screen_batch = [d[3] for d in minibatch]

            readout_j1_batch = network.eval(feed_dict={x: screen_batch})
            y_batch = []
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(reward_batch[i] + 0.99 * np.max(readout_j1_batch[i]))

            train_step.run(feed_dict={y: y_batch, a: action_batch, x: s_j_batch})

        if step % 1000 == 0:
            saver.save(sess, 'saved_networks/pacman-dqn', global_step=step)

        s_t = image_data
        step += 1
        print("Finish step : ", step)

if __name__ == '__main__':
    train_network()