import numpy as np

class KMeans:
    def __init__(self, data, num_class):
        self.data = data
        self.num_class = num_class

    def train(self, max_iterations):
        # 1. 先随机选择 K 个质心 
        center = KMeans.center_init(self.data, self.num_class)
        # 2. 开始训练
        num_examples = self.data.shape[0]
        closest_center_ids = np.empty((num_examples, 1))
        for _ in range(max_iterations):
            # 3. 得到当前每一个样本点到 K 个中心点的距离，找到最近的
            closest_center_ids = KMeans.center_find_closest(self.data, center)
            # 4. 进行中心点位置更新
            center = KMeans.center_compute(self.data, closest_center_ids, self.num_class)
        return center, closest_center_ids
        

    @staticmethod
    def center_init(data, num_class):
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)
        center = data[random_ids[:num_class],:]
        return center

    @staticmethod
    def center_find_closest(data, center):
        num_examples = data.shape[0]
        num_centers = center.shape[0]
        closest_center_ids = np.zeros((num_examples, 1))
        for example_index in range(num_examples):
            distance = np.zeros((num_centers, 1))
            for center_index in range(num_centers):
                distance_diff = data[example_index, :] - center[center_index, :]
                distance[center_index] = np.sum(distance_diff**2)
            closest_center_ids[example_index] = np.argmin(distance)
        return closest_center_ids


    @staticmethod
    def center_compute(data, closest_center_ids, num_class):
        num_features = data.shape[1]
        center = np.zeros((num_class, num_features))
        for center_id in range(num_class):
            closest_id = closest_center_ids == center_id
            center[center_id] = np.mean(data[closest_id.flatten(), :], axis=0)
        return center
