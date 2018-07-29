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

def test_softmax_regression():
    ## 训练集用于训练，校验集用于判断训练停止的条件，测试集用于检测训练效果。 
    input = mnist.input_data.read_data_sets("MNIST_data/", one_hot=True) # mnist数据集
    print '训练集数据和标签数量、维度:', input.train.images.shape, input.train.labels.shape
    print '测试集数据和标签数量、维度:', input.test.images.shape, input.test.labels.shape
    print '验证集数据和标签数量、维度:', input.validation.images.shape, input.validation.labels.shape
    
    
    # 预测公式
    x = tf.placeholder(tf.float32, [None, 784]) # 占位符, 表示一个待输入任意数量的784维向量
    W = tf.Variable(tf.zeros([784, 10])) # 张量, 可修改的变量 
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b) # 定义模型
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # 损失函数

    # 设置优化算法
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.initialize_all_variables() # 初始化创建的变量
    sess = tf.Session()
    sess.run(init) # 启动模型



    #sess = tf.InteractiveSession() # 将当前创建的seesion注册为默认session
    ## 全局参数初始化器初始化参数
    #tf.global_variables_initializer().run()
    
    # 训练
    for _ in range(1000):
        batch_xs, batch_ys = input.train.next_batch(100) # 随机抽取100个样本(图片, 标签)
        #train_step.run({x: batch_xs, y_: batch_ys})
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) 

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print accuracy.eval({x: input.test.images, y_: input.test.labels})
    print sess.run(accuracy, feed_dict={x: input.test.images, y_: input.test.labels})

if __name__ == "__main__":
    test_softmax_regression()

