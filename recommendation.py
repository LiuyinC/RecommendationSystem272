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
    """
    Read in test data and return a dictionary of user samples
    """
    data_file = urllib2.urlopen(test_url)
    # data = data_file.readlines()
    users = {}
    for user_sample in data_file.readlines():
        [userid, movie, score] = map(int, user_sample.split())
        if userid in users.keys():
            if score > 0:
                users[userid].add_rated_movie(movie, score)
            else:
                users[userid].add_prediction(movie)
        else:
            users[userid] = TestUser(userid)
            assert score > 0, "User did not give rate!"
            users[userid].add_rated_movie(movie, score)
    return users


class TestUser:
    """
    A simple class for test sample
    """
    def __init__(self, userid):
        """
        initial user sample with empty list of rated movies and empty dictionary of predicting movies
        """
        self._userid = userid
        self._rated_movies = {}
        self._predictions = {}


    def get_userid(self):
        """
        :return: user id
        """
        return self._userid


    def get_rated_movies(self):
        """
        Return rated movies as a dictionary
        """
        return self._rated_movies


    def get_predictions(self):
        """
        Return predicted movies as a dictionary
        """
        return self._predictions


    def add_rated_movie(self, movie_id, movie_score):
        """
        Add a new rated movie into rated_movie dictionary
        """
        assert movie_id not in self._rated_movies.keys(), "This movie is already rated."
        self._rated_movies[movie_id] = movie_score


    def add_prediction(self, movie_id):
        """
        Add the movie id into prediction dictionary and initial its score as zero
        """
        assert movie_id not in self._rated_movies.keys(), "This movie has been rated"
        assert movie_id not in self._predictions.keys(), "This movie is already in prediction list"
        self._predictions[movie_id] = 0


# data = read_train(TRAINING_URL, TRAINING_USERS, TRAINING_MOVIES)
# print data[199]
# print data[199][-17]
# data2 = data.split('\r')
# print data2
# print len(data2)

# testdata = read_test(TESTING5_URL)

# print "rated movies", testdata[201].get_rated_movies()
# print "predictions", testdata[201].get_predictions()

# l = '201 237 4\r\n'
# a = map(int, l.split())
# print a, len(a), type(a)