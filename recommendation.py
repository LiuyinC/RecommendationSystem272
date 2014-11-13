__author__ = 'liuyincheng'

"""
Develop different algorithms to make recommendations for movies.
"""

import urllib2
import numpy as np


TRAINING_URL = 'http://www.cse.scu.edu/~yfang/coen272/train.txt'
TESTING5_URL = 'http://www.cse.scu.edu/~yfang/coen272/test5.txt'
TESTING10_URL = 'http://www.cse.scu.edu/~yfang/coen272/test10.txt'
TESTING20_URL = 'http://www.cse.scu.edu/~yfang/coen272/test20.txt'


def read_train(train_url):
    """
    Read training data from internet and return a nested dictionary in which each element is a training sample.
    Each training sample contains rating scores on 1000 movies.
    Score 0 means that the user does not rate the movie.
    """
    data_file= urllib2.urlopen(train_url)
    data = data_file.readline()
    data = data.split()
    user_movie = {}
    for userid in range(200, 0, -1):
        user_movie[userid] = {}
        for movieid in range(1000, 0, -1):
            user_movie[userid][movieid] = int(data.pop())
    return user_movie


def read_test(test_url):
    data_file = urllib2.urlopen(test_url)
    data = data_file.readline()
    print data
    # print len(data)
    return data


# data = read_test(TRAINING_URL)
# print type(data)
# data2 = data.split('\r')
# print data2
# print len(data2)


# l1 = [1,2,3,4,5,6]
# l2 = np.asarray(l1)
# l3 = np.reshape(l2, (3,2))
# print l1
# print l2
# print l3

# s1 = ['1', '2', '3', '4']
# l1 = map(int, s1)
# print l1