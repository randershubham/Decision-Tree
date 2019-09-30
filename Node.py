import trainDT
import numpy as np


class Node:
    def __init__(self, data_idx, impurity_method, node_level, impurity_value):
        self.data_idx = data_idx
        self.impurity_method = impurity_method
        self.impurity_value = impurity_value
        self.node_level = node_level

        # defaults
        self.dfeature = -1
        self.nfeatures = []
        self.majority_class = -1
        self.left_child = None
        self.right_child = None

    @staticmethod
    def _init_node(data_idx, impurity_method, level, impurity_value):
        return Node(data_idx, impurity_method, level, impurity_value)

    def get_label_counts(self, label_list):
        labels = [1, 2, 3, 4, 5]
        count_of_labels = []
        for label in labels:
            count_of_labels.append(len(label_list[label_list == label]))
        return count_of_labels

    def build_decision_tree(self, data, indices, impurity_method, nl, p_threshold):
        _impurity_value = -1
        labels = trainDT.train_y
        count_of_labels = self.get_label_counts(labels)
        if self.impurity_method == "gini":
            _impurity_value = self.gini_index(count_of_labels)
        elif self.impurity_method == "entropy":
            _impurity_value = self.entropy(count_of_labels)
        else:
            raise ("Invalid impurity method provided " + str(self.impurity_method))

        print(_impurity_value)

        decision_tree = Node._init_node(indices,
                                        impurity_method=impurity_method,
                                        level=nl,
                                        impurity_value=_impurity_value)
        decision_tree.split_node(max_levels=nl, p_threshold=p_threshold)
        return decision_tree

    def split_node(self, max_levels, p_threshold):
        if self.node_level > max_levels and self.impurity_value > p_threshold:
            max_gain = -1
            split_feature = -1
            final_left_indices = []
            final_right_indices = []
            final_left_impurity = -1
            final_right_impurity = -1

            for feature in self.nfeatures:
                left_indices = self.get_indices_for_feature(self.data_idx, feature, 0)
                right_indices = self.get_indices_for_feature(self.data_idx, feature, 1)

                p_left = self.calculate_ip(self.data_idx)
                p_right = self.calculate_ip(self.data_idx)

                m = self.get_weighted_impurity(p_left, p_right)
                gain = self.impurity_value - m

                if gain > max_gain:
                    split_feature = feature
                    max_gain = gain
                    final_left_indices = left_indices
                    final_right_indices = right_indices
                    final_left_impurity = p_left
                    final_right_impurity = p_right

            self.dfeature = split_feature
            self.left_child = self._init_node(final_left_indices,
                                              impurity_method=self.impurity_method,
                                              level=self.node_level + 1,
                                              impurity_value=final_left_impurity)

            self.right_child = self._init_node(final_right_indices,
                                               impurity_method=self.impurity_method,
                                               level=self.node_level + 1,
                                               impurity_value=final_right_impurity)

            self.right_child.split_node(max_levels, p_threshold)
            self.left_child.split_node(max_levels, p_threshold)

    def get_weighted_impurity(self, impurity_left, impurity_right, count_left, count_right):
        sum = count_left + count_right
        return impurity_left * (count_left / sum) + impurity_right * (count_right / sum)

    def get_indices_for_feature(self, indices, feature_num, expected_value):
        final_index_set = []
        for index in indices:
            feature_value = trainDT.train_x[index, feature_num]
            if feature_value is expected_value:
                final_index_set.append(index)
        return final_index_set

    def calculate_ip(self, indices):

        final_indexed_labels = trainDT.train_y[indices]
        count_of_each_labels = self.get_label_counts(final_indexed_labels)

        _impurity_value = -1
        if self.impurity_method is "gini":
            _impurity_value = self.gini_index(count_of_each_labels)
        elif self.impurity_method is "entropy":
            _impurity_value = self.entropy(count_of_each_labels)
        else:
            raise ("Invalid impurity method provided: " + str(self.impurity_method))

        return _impurity_value

    def gini_index(self, counts):
        counts = np.array(counts)
        print(counts)
        print(np.sum(counts))
        probabilities = np.divide(counts, np.sum(counts))
        sum_probabilities = np.sum(np.power(probabilities, 2))
        return 1 - sum_probabilities

    def entropy(self, counts):
        counts = np.array(counts)
        probabilities = np.divide(counts, np.sum(counts))
        entropy = -probabilities * (np.log(probabilities) / np.log(2))
        return entropy
