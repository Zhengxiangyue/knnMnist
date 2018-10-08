import numpy
import pickle
import os
import sys
import json
from threading import Thread
from collections import Counter
from multiprocessing import Process
import matplotlib.pyplot as plt

if __name__ == '__main__':

    def pre_process(data_set):
        """
        normalize image pixel to 0 and 1
        :param data_set:
        :return:
        """
        return [[0 if i < 80 else 1 for i in each] for each in data_set]


    def predict(train_images, test_image, labels, k):
        """
        train_set and data both contain only 0 and 1, use (train_set[each] - data).count(1) as distance
        :param train_set:
        :param data:
        :return:
        """
        diff_array = []
        for index, image in enumerate(train_images):
            diff_array.append([bin(image ^ test_image).count('1'), labels[index]])
        votes = [i[1] for i in sorted(diff_array)[:k]]
        return Counter(votes).most_common(1)[0][0]

    def single_digit(test_data, test_labels, digit):
        test_data = [each for index, each in enumerate(test_data) if test_labels[index] == digit]
        test_labels = [digit] * len(test_data)
        return test_data, test_labels

    def run():
        path = './data/train.csv'
        data_cache = path + '.data_cache'
        labels_cache = path + '.labels_cache'
        train_rate = 0.90

        # data loading
        if not os.path.isfile(data_cache):
            print(">> Loading data from file...")
            all_data = numpy.genfromtxt(path, delimiter=',', skip_header=1).astype(numpy.dtype('int')).tolist()
            data, labels = pre_process([each[1:] for each in all_data]), [each[0] for each in all_data]

            bit_array = [int("".join(str(x) for x in image), 2) for image in data]

            with open(data_cache, 'wb') as f:
                pickle.dump(bit_array, f)
            with open(labels_cache, 'wb') as f:
                pickle.dump(labels, f)
        else:
            print(">> Loading data from cache...")
            with open(data_cache, 'rb') as f:
                bit_array = pickle.load(f)
            with open(labels_cache, 'rb') as f:
                labels = pickle.load(f)

        # divide data into train set and test set, based on the ratio
        dividing_index = int(len(bit_array) * train_rate)
        train_data, train_labels = bit_array[:dividing_index], labels[:dividing_index]
        test_data, test_labels = bit_array[dividing_index:], labels[dividing_index:]

        test_data, test_labels = single_digit(test_data, test_labels, 9)
        print(">> Total %s test data" % len(test_data))

        for k in range(9, 21, 2):
            print(">> k is %s" % k)
            confusion_matrix = [[0 for x in range(10)] for y in range(10)]
            for index, each in enumerate(test_data):
                pred, real = predict(train_data, each, train_labels, k), test_labels[index]
                confusion_matrix[pred][real] += 1
            print float(confusion_matrix[9][9]) / len(test_data)



    run()


