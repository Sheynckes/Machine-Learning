# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 23:40
# @Author  : SHeynckes
# @Email   : sheynckes@outlook.com
# @File    : Perceptron.py
# @Software: PyCharm


# 使用Python实现感知器学习算法

import numpy as np


class PerceptronSimple(object):
    # 通过使用面向对象编程的方式在一个Python类中定义感知器的接口，使得我们可以初始化新的感知器对象
    # 使用类中定义的fit()方法对感知器进行训练，使用predict()方法利用训练后的感知器进行预测
    """
    Perceptron classifier.

    Parameters
    ----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes over the training dataset

    Attributes
    ----------
    w_: 1d-array
        Weights after fitting
    errors_: list
        Number of misclassifications in every epoch.
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X: {array-like}, shape=[n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y: array-like, shape=[n_samples]
            Target values:

        Returns
        ----------
        self: object
        """
        # 将权重系数初始化为0
        # 按照Python开发的惯例，对于那些并非再初始化对象时创建、但是又被对象中其他方法调用的属性，
        # 可以在其后面添加一个下划线，例如：self.w_。
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # 更新权重系数
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                # 累计错分类的数量
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate net input
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
