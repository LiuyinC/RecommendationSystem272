__author__ = 'liuyincheng'

"""
Develop different algorithms to make recommendations for movies.
"""

import urllib2
import numpy as np


TRAINING_URL = 'http://www.cse.scu.edu/~yfang/coen272/train.txt'
TRAINING_USERS = 200
TRAINING_MOVIES = 1000
TESTING5_URL = 'http://www.cse.scu.edu/~yfang/coen272/test5.txt'
TESTING10_URL = 'http://www.cse.scu.edu/~yfang/coen272/test10.txt'
TESTING20_URL = 'http://www.cse.scu.edu/~yfang/coen272/test20.txt'


def read_train(train_url, num_user, num_movie):
    """
    Read training data from internet and return an array matrix, in which row represents user,
    and column represents movie rate score.
    Score 0 means that the user does not rate the movie.
    """
    data_file= urllib2.urlopen(train_url)
    data = data_file.readline()
    data_array = np.asarray(map(int, data.split()))
    data_matrix = np.reshape(data_array, (num_user, num_movie))
    return data_matrix


def read_test(test_url):
    data_file = urllib2.urlopen(test_url)
    data = data_file.readline()
    print data
    # print len(data)
    return data


data = read_train(TRAINING_URL, TRAINING_USERS, TRAINING_MOVIES)
# print data[199]
# print data[199][-17]
# data2 = data.split('\r')
# print data2
# print len(data2)


# l1 = [1,2,3,4,5,6]
# l2 = np.asarray(l1)
# l3 = np.reshape(l2, (3,2))
# print l1
# print l2
# print l3
