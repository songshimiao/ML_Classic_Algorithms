'''
线性回归
'''
import numpy as np
from prepare_for_training import prepare_for_training


class LinearRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        '''
        1. 对数据进行预处理操作
        2. 得到所有的特征个数
        3. 初始化参数矩阵
        '''
        (data_processed, features_mean, features_deviation) = prepare_for_training(
            data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, lr, num_iterations=500):
        '''
        训练模块，执行梯度下降
        '''
        loss_history = self.gradient_descent(lr, num_iterations)
        return self.theta, loss_history

    def gradient_descent(self, lr, num_iterations):
        '''
        实际迭代模块
        '''
        loss_history = []
        for _ in range(num_iterations):
            self.gradient_step(lr)
            loss_history.append(self.loss_function(self.data, self.labels))
        return loss_history

    def gradient_step(self, lr):
        '''
        梯度下降参数更新计算方法
        '''
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - lr*(1/num_examples)*(np.dot(delta.T, self.data)).T
        self.theta = theta

    def loss_function(self, data, labels):
        '''
        损失计算方法
        '''
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels
        loss = (1/2)*np.dot(delta.T, delta)/num_examples
        return loss[0][0]

    @staticmethod
    def hypothesis(data, theta):
        predictions = np.dot(data, theta)
        return predictions

    def get_loss(self, data, labels):
        data_processed = prepare_for_training.prepare_for_training(
            data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]

        return self.loss_function(data_processed, labels)

    def predict(self, data):
        '''
        用训练好的参数模型，去预测得到回归结果
        '''
        data_processed = prepare_for_training(
            data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]

        predicted = LinearRegression.hypothesis(data_processed, self.theta)
        return predicted
