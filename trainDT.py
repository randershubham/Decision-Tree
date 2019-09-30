import argparse
import numpy as np


class Node:

    def __init__(self, data_idx, impurity_method, node_level, impurity_value, nfeatures):
        self.data_idx = data_idx
        self.impurity_method = impurity_method
        self.impurity_value = impurity_value
        self.node_level = node_level
        self.nfeatures = nfeatures

        # defaults
        self.dfeature = -1
        self.majority_class = -1
        self.left_child = None
        self.right_child = None

    @staticmethod
    def _init_node(data_idx, impurity_method, level, impurity_value, nfeatures):
        return Node(data_idx, impurity_method, level, impurity_value, nfeatures)

    @staticmethod
    def get_label_counts(actual_labels_list):
        labels = set(train_y)
        count_of_labels = []
        for label in labels:
            count = len(actual_labels_list[actual_labels_list == label])
            if count != 0:
                count_of_labels.append(count)
        return count_of_labels

    @staticmethod
    def get_weighted_impurity(impurity_left, impurity_right, count_left, count_right):
        total_sum = count_left + count_right
        return impurity_left * (count_left / total_sum) + impurity_right * (count_right / total_sum)

    @staticmethod
    def get_indices_for_feature(indices, feature_num, expected_value):
        final_index_set = []
        for index in indices:
            feature_value = train_x[index, feature_num]
            if feature_value == expected_value:
                final_index_set.append(index)
        return final_index_set

    @staticmethod
    def buildDT(_data, _indices, _impurity_method, _nl, _p_threshold):
        _initial_impurity_value = -1

        if _impurity_method == "gini":
            _initial_impurity_value = Node.calculateGINI(_indices)
        elif _impurity_method == "entropy":
            _initial_impurity_value = Node.calculateEntropy(_indices)
        else:
            raise ("Invalid impurity method provided " + str(_impurity_method))

        _decision_tree = Node._init_node(_indices,
                                         impurity_method=_impurity_method,
                                         level=0,
                                         impurity_value=_initial_impurity_value,
                                         nfeatures=range(_data.shape[1]))
        _decision_tree.split_node(max_levels=_nl, _p_threshold=_p_threshold)
        return _decision_tree

    @staticmethod
    def calculateGINI(indices):
        if len(indices) == 0:
            return 0
        final_indexed_labels = train_y[indices]
        count_of_each_labels = Node.get_label_counts(final_indexed_labels)
        counts = np.array(count_of_each_labels)
        probabilities = np.divide(counts, np.sum(counts))
        sum_probabilities = np.sum(np.power(probabilities, 2))
        return 1 - sum_probabilities

    @staticmethod
    def calculateEntropy(indices):
        if len(indices) == 0:
            return 0
        final_indexed_labels = train_y[indices]
        count_of_each_labels = Node.get_label_counts(final_indexed_labels)
        counts = np.array(count_of_each_labels)
        probabilities = np.divide(counts, np.sum(counts))
        entropy = np.sum(-probabilities * (np.log(probabilities) / np.log(2)))
        return entropy

    @staticmethod
    def get_maximum_occurring_label(indices):
        labels = set(train_y)
        actual_labels_list = train_y[indices]
        max_occurring_label = -1
        max_count = -1
        for label in labels:
            count = len(actual_labels_list[actual_labels_list == label])
            if count > max_count:
                max_count = count
                max_occurring_label = label

        return max_occurring_label

    def split_node(self, max_levels, _p_threshold):
        self.majority_class = Node.get_maximum_occurring_label(self.data_idx)
        if self.node_level < max_levels and self.impurity_value > _p_threshold:
            max_gain = 0
            split_feature = -1
            final_left_indices = []
            final_right_indices = []
            final_left_impurity = -1
            final_right_impurity = -1

            for feature in self.nfeatures:
                left_indices = self.get_indices_for_feature(self.data_idx, feature, 0)
                right_indices = self.get_indices_for_feature(self.data_idx, feature, 1)

                p_left = self.calculate_ip(left_indices)
                p_right = self.calculate_ip(right_indices)

                total_sum = len(left_indices) + len(right_indices)
                m = p_left * (len(left_indices) / total_sum) + p_right * (len(right_indices) / total_sum)
                gain = self.impurity_value - m

                if gain > max_gain:
                    split_feature = feature
                    max_gain = gain
                    final_left_indices = left_indices
                    final_right_indices = right_indices
                    final_left_impurity = p_left
                    final_right_impurity = p_right

            self.dfeature = split_feature
            # print(split_feature)
            # print(max_gain)
            # print("----")

            if len(final_left_indices) > 0:
                self.left_child = self._init_node(final_left_indices,
                                                  impurity_method=self.impurity_method,
                                                  level=self.node_level + 1,
                                                  impurity_value=final_left_impurity,
                                                  nfeatures=self.nfeatures)
                self.left_child.split_node(max_levels, p_threshold)

            if len(final_right_indices) > 0:
                self.right_child = self._init_node(final_right_indices,
                                                   impurity_method=self.impurity_method,
                                                   level=self.node_level + 1,
                                                   impurity_value=final_right_impurity,
                                                   nfeatures=self.nfeatures)

                self.right_child.split_node(max_levels, p_threshold)

    def calculate_ip(self, indices):
        if self.impurity_method == "gini":
            return self.calculateGINI(indices)
        elif self.impurity_method == "entropy":
            return self.calculateEntropy(indices)
        else:
            raise ("Invalid impurity method provided: " + str(self.impurity_method))

    def classify(self, _test_x, _output_file):
        _file = open(_output_file, "w")
        for record in _test_x:
            _predicted_label = self.get_predicted_label(record)
            _file.write(str(_predicted_label))
            _file.write("\n")
        _file.close()

    def get_predicted_label(self, _test_x_record):
        if self.right_child is None and self.left_child is None:
            return self.majority_class
        else:
            distinguishing_feature_value = _test_x_record[self.dfeature]
            if distinguishing_feature_value == 0:
                return self.left_child.get_predicted_label(_test_x_record)
            else:
                return self.right_child.get_predicted_label(_test_x_record)


def load_data_file_and_label(data_file, label_file):
    data = np.genfromtxt(data_file).astype(int)
    label = np.genfromtxt(label_file).astype(int)
    return data, label


def get_parser():
    _parser = argparse.ArgumentParser()

    _parser.add_argument('-train_data')
    _parser.add_argument('-train_label')
    _parser.add_argument('-test_data')
    _parser.add_argument('-test_label')
    _parser.add_argument('-nlevels')
    _parser.add_argument('-pthrd')
    _parser.add_argument('-impurity')
    _parser.add_argument('-pred_file')

    return _parser


def get_accuracy(expected, predicted):
    count = 0
    for i in range(0, len(expected)):
        if expected[i] == predicted[i]:
            count = count + 1
    return count / len(expected)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    train_data_file = str(args.train_data)
    train_label_file = str(args.train_label)
    test_data_file = str(args.test_data)
    test_label_file = str(args.test_label)
    max_level_input = int(args.nlevels)
    p_threshold = float(args.pthrd)
    impurity = str(args.impurity)
    pred_output_file = str(args.pred_file)

    train_x, train_y = load_data_file_and_label(train_data_file, train_label_file)
    test_x, test_y = load_data_file_and_label(test_data_file, test_label_file)

    decision_tree = Node.buildDT(train_x,
                                 _indices=list(range(400)),
                                 _impurity_method=impurity,
                                 _nl=max_level_input,
                                 _p_threshold=p_threshold)

    print(Node.calculateGINI(range(0, 400)))

    decision_tree.classify(test_x, _output_file=pred_output_file)
    print(pred_output_file)
    prediction_file = open(pred_output_file, "r")
    predictions = np.genfromtxt(prediction_file)

    print(get_accuracy(test_y, predictions))
