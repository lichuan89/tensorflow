#!/usr/bin/env python
## -*- coding:utf8 -*-

"""
    @author licuha89@126.com
    @date   2016/09/12  
    @note
"""

import sys
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def test_softmax_regression():

    # 训练集、验证集、测试集
    input = mnist.input_data.read_data_sets("MNIST_data/", one_hot=True) # mnist数据集
    print '训练集数据和标签数量、维度:', input.train.images.shape, input.train.labels.shape
    print '测试集数据和标签数量、维度:', input.test.images.shape, input.test.labels.shape
    print '验证集数据和标签数量、维度:', input.validation.images.shape, input.validation.labels.shape
    
    
    # 预测模型 
    x = tf.placeholder(tf.float32, [None, 784]) # 占位符, 表示一个待输入任意数量的784维向量
    W = tf.Variable(tf.zeros([784, 10])) # 张量, 可修改的变量 
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b) # 定义模型
    y_ = tf.placeholder(tf.float32, [None, 10])


    # 输入矩阵为 [-1, 28,28, 1]
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # 5x5x1x32的矩阵: 32个不同卷积核, 5x5尺寸, 1个颜色通道
    W_conv1 = weight_variable([5, 5, 1, 32]) 
    b_conv1 = bias_variable([32]) # 偏置
    # 卷积 && Relu 激活, 结果矩阵为[-1, 28, 28, 32]
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # 2x2最大池化,步长为2, 结果矩阵为[-1, 14, 14, 32]
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    # 卷积 && Relu 激活, 结果矩阵为[-1, 14, 14, 64]
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # 池化, 结果矩阵为[-1, 7， 7， 64】
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    # 全连接层, 输出[-1, 1024] 
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    # dropout层, 以keep_prob概率随机丢弃一部分节点进行训练, 输出[-1, 1024] 
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    # sofmax层, 输出概率矩阵[-1, 10] 
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    # 交叉熵损失函数
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

    # Adam优化器
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    #  准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    print 'step\taccuracy'
    for i in range(20000):
        # 大小为50的mini-batch进行20000次迭代
        batch = input.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
            print '%d\t%g' % (i, train_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print "test accuracy %g"%accuracy.eval(feed_dict={
            x: input.test.images, y_: input.test.labels, keep_prob: 1.0})


if __name__ == "__main__":
    test_softmax_regression()
