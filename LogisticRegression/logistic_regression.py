import numpy as np
from scipy.optimize import minimize
from LinearRegression import prepare_for_training
from LogisticRegression import sigmoid


class LogisticRegression:
    def __init__(self, data, labels, normalize_data=True):
        '''
        1. 对数据进行预处理操作
        2. 得到所有的特征个数
        3. 初始化参数矩阵
        '''
        (data_processed, features_mean, features_deviation) = prepare_for_training(
            data, normalize_data=True)

        self.data = data_processed
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        num_unique_labels = np.unique(labels).shape[0]
        self.theta = np.zeros((num_unique_labels, num_features))

    def train(self, max_iteration=1000):
        loss_history = []
        num_features = self.data.shape[1]
        for label_index, unique_label in enumerate(self.unique_labels):
            current_initial_theta = np.copy(
                self.theta[label_index].reshape(num_features, 1))
            current_labels = (self.labels == unique_label).astype(float)
            (current_theta, loss_history) = LogisticRegression.gradient_descent(
                self.data, current_labels, current_initial_theta, max_iteration)

    @staticmethod
    def gradient_descent(data, labels, current_initial_theta, max_iteration):
        loss_history = []
        num_features = data.shape[1]
        result = minimize(
            # 要优化的目标
            lambda current_theta: LogisticRegression.loss_function(
                data, labels, current_initial_theta.reshape(num_features, 1)),
            # 初始化权重参数
            current_initial_theta,
            # 选择优化策略
            method='CG',
            # 梯度下降迭代计算公式
            jac=lambda current_theta: LogisticRegression.gradient_step(
                data, labels, current_initial_theta.reshape(num_features, 1)),
            # 记录结果
            callback=lambda current_theta: loss_history.append(LogisticRegression.loss_function(
                data, labels, current_theta.reshape(num_features, 1))),
            # 迭代次数
            options={'maxiter': max_iteration}
        )

        if not result.success:
            raise ArithmeticError('Can not minimize loss function'+ result.message)
        
        optimized_theta = result.x.reshape(num_features, 1)
        return optimized_theta, loss_history

    @staticmethod
    def loss_function(data, labels, theta):
        num_examples = data.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)
        y_is_set_loss = np.dot(
            labels[labels == 1].T, np.log(predictions[labels == 1]))
        y_is_not_loss = np.dot(
            labels[labels == 0].T, np.log(1-predictions[labels == 1]))
        loss = (-1/num_examples)*(y_is_set_loss+y_is_not_loss)

        return loss

    @staticmethod
    def hypothesis(data, theta):
        predictions = sigmoid(np.dot(data, theta))
        return predictions

    @staticmethod
    def gradient_step(data, labels, theta):
        num_examples = labels.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)
        label_diff = predictions - labels
        gradients = (1/num_examples)*np.dot(data.T, label_diff)

        return gradients.T.flatten()


    def predict(self, data):
        num_examples = data.shape[0]
        data_processed = prepare_for_training(
            data, normalize_data=True)[0]

        prob = LogisticRegression.hypothesis(data_processed, self.theta.T)
        max_prob_index = np.argmax(prob, axis=1)
        class_prediction = np.empty(max_prob_index.shape, dtype=object)
        for index, label in enumerate(self.unique_labels):
            class_prediction[max_prob_index == index] = label

        return class_prediction.reshape((num_examples, 1))
