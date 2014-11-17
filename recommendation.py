__author__ = 'liuyincheng'

"""
Develop different algorithms to make recommendations for movies.
"""

import urllib2
import numpy as np
from scipy import linalg


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


    def update_prediction(self, movie_id, prediction):
        """
        Update prediction score
        """
        assert self._predictions[movie_id] == 0, "This movie has been predicted before"
        self._predictions[movie_id] = prediction


    def output_predictions(self, output_file):
        """
        Write predictions as required format into file, the file has been opened.
        """
        predictions = self.get_predictions()
        for movie in sorted(predictions.keys()):
            output_file.write(str(self.get_userid()) + "\t" + str(movie) + "\t" + str(predictions[movie]) + "\r")


    def user_based_CF(self, predict_movie, training_data, num_nearest_neighbors, similarity_method):
        """
        Get other rating information from training data where rating score of predict movie is not zero.
        If user_based is true, the rating information only contains test user's rated movies.
        And return a n * (m+1) matrix, where n is number of relevant neighbors, and m is number of rated movies,
        and the last column is rated scores of predicting movie.
        If user_based is false, the rating information ?????????

        """
        def cosine_similarity(vect2, vect1):
            """
            Compute the basic cosine similarity between vect1 and vect2 with the same dimensions.
            vect1 can't contain zero value, is the test user's rating.
            Vect1 and vect2 are np.arrays
            """
            assert sum(vect1 <= 0) == 0, "Test User's rating score has zero value"
            assert np.size(vect1) == np.size(vect2), "Don't have the same dimension"
            non_zeros = vect2 > 0
            if sum(non_zeros) == 0:
                # vector2 has no rated scores
                cosine = 0.0
            elif sum(non_zeros) == 1:
                # fix the issue that if vectors have dimension of 1, their cosine similarity is always 1
                cosine = 0.8 - float(abs(vect1[non_zeros] - vect2[non_zeros])) * 0.2
            else:
                vector1 = vect1[non_zeros].astype('float')
                vector2 = vect2[non_zeros].astype('float')
                cosine = sum(vector1 * vector2) / np.sqrt(sum(vector1 * vector1) * sum(vector2 * vector2))
            return cosine


        def pearson_correlation(vect2, vect1):
            """
            Calculate the pearson_correlation of vect1 and vect2 and
            return the weights.
            Vect1 represents the test user and can't have zero element(s).
            Last two values of vect2 are predicting movie's rating and neighbor's bias
            """
            assert sum(vect1 <= 0) == 0, "Test User has zero rating."
            assert np.size(vect1) == np.size(vect2) - 2, "Don't have the same dimension"
            non_zeros = vect2[0:-2] > 0
            vector1 = vect1[non_zeros].astype('float')
            vector2 = vect2[non_zeros].astype('float')
            vect1_ave = np.average(vect1)
            vect2_ave = vect2[-1]


            if np.size(np.nonzero(vector2 - vect2_ave)) == 0 or sum(non_zeros) == 0:
                # The training use has no preference on any movies OR it has no rating.
                return 0.0

            if sum(non_zeros) == 1:
                # Fix the issue that if vectors have dimension of 1, their cosine similarity is always 1
                weight = 0.8 - float(abs(vect1[non_zeros] - vect2[non_zeros])) * 0.2
                return weight

            if np.size(np.nonzero(vector1 - vect1_ave)) == 0:
                # Test user has no preference on such data, which movies are also rated by neighbors
                return cosine_similarity(vect2[0:-2], vect1)

            weight = sum((vector2 - vect2_ave) * (vector1 - vect1_ave)) / np.sqrt(sum((vector1 - vect1_ave)*(vector1 - vect1_ave) ) * sum((vector2 - vect2_ave)*(vector2 - vect2_ave)))
            assert -1.0001 < weight < 1.0001, "Weight " + str(weight) +" is out of range\n" +"training: " + str(vect2) + " Test user: " + str(vect1) + "\nvectors " + str(vector2) + str(vector1) + "\n average" + str(vect2_ave) + str(vect1_ave)
            return weight


        # Get relevant neighbors as a n * (m +1) matrix, where n is the number of neighbors,
        # first m columns are relevant movies and last column is predicting movie's score.
        rated_movies = self.get_rated_movies()
        rated_movies_indices = np.array(sorted(rated_movies.keys())) - 1
        predict_movie_index = predict_movie - 1
        predict_non_zero_id = np.nonzero(training_data[:, predict_movie_index])[0]
        relevant_neighbors = training_data[predict_non_zero_id , :][:, np.hstack((rated_movies_indices, predict_movie_index))]

        # Get test user's rating as a np.array
        user_rated_scores = []
        for key in sorted(rated_movies.keys()):
            user_rated_scores.append(rated_movies[key])
        user_rated_scores = np.array(user_rated_scores)

        # If the user doesn't have any preference on any movies, or this movie has not been rated by any other users,
        # Set prediction as the average of user's rating.
        if np.size(np.nonzero(user_rated_scores - np.average(user_rated_scores))) == 0 or np.size(relevant_neighbors) == 0:
            prediction = int(np.average(user_rated_scores))
            self.update_prediction(predict_movie, prediction)
            return

        # Add bias to last column
        masked_neighbors = np.ma.masked_equal(training_data[predict_non_zero_id, :], 0)
        neighbor_bias = np.ma.average(masked_neighbors, 1)
        relevant_neighbors = np.hstack((relevant_neighbors, neighbor_bias.reshape((len(neighbor_bias), 1))))
        # print relevant_neighbors
        if similarity_method == 'Cosine':
            """
            Predict movie by cosine similarity
            """
            # Calculate similarities
            similarities = np.apply_along_axis(cosine_similarity, 1, relevant_neighbors[:, 0:-2], user_rated_scores)
            # Find k nearest neighbors
            k_sorted_indices = np.argsort(similarities)[::-1][0:num_nearest_neighbors]

            # Calculate prediction
            if sum(similarities[k_sorted_indices]) == 0:
                init_prediction = np.average(relevant_neighbors[:, -2][k_sorted_indices])
            else:
                init_prediction = sum(similarities[k_sorted_indices] * relevant_neighbors[:, -2][k_sorted_indices]) / sum(similarities[k_sorted_indices])
            # print similarities[k_sorted_indices], k_sorted_indices, np.size(relevant_neighbors), relevant_neighbors
            assert 0 < init_prediction < 5.0001, "Prediction " + str(init_prediction) +" is out of reasonable range (0, 5]"
            if init_prediction < 1:
                prediction = 1
            else:
                prediction = int(round(init_prediction))

            self.update_prediction(predict_movie, prediction)

        elif similarity_method == 'Pearson':
            """
            Predict movie by pearson correlation.
            """
            # Calculate weights
            weights = np.apply_along_axis(pearson_correlation, 1, relevant_neighbors, user_rated_scores)
            # Find the k nearest neighbors
            k_sorted_indices = np.argsort(np.absolute(weights))[::-1][0:num_nearest_neighbors]
            # Calculate testing user's average base
            user_average = np.average(user_rated_scores)

            # Calculate prediction
            if sum(np.absolute(weights[k_sorted_indices])) == 0:
                init_prediction = user_average
            else:
                neighbor_natural_ratings = relevant_neighbors[:,-2] - relevant_neighbors[:, -1]
                # print "weights", weights
                # print "k weights", weights[k_sorted_indices]
                init_prediction = user_average + sum(weights[k_sorted_indices] * neighbor_natural_ratings[k_sorted_indices]) / sum(np.absolute(weights[k_sorted_indices]))
                # print sum(weights[k_sorted_indices] * neighbor_natural_ratings[k_sorted_indices]) / sum(np.absolute(weights[k_sorted_indices]))

            assert -5.000 < init_prediction < 10.000, "Prediction " + str(init_prediction) +" is out of reasonable range (0, 5]"

            if init_prediction < 1:
                prediction = 1
            elif init_prediction > 5:
                prediction = 5
            else:
                prediction = int(round(init_prediction))

            self.update_prediction(predict_movie, prediction)

        else:
            print "Wrong similarity method input"
            return None


def run_example():
    """
    Run the recommendation system
    """
    training_data = read_train(TRAINING_URL, TRAINING_USERS, TRAINING_MOVIES)
    test_users = read_test(TESTING5_URL)
    output_file = open("result5_ub_pearson_20nbs.txt", "w")
    for user in test_users.values():
        for predicting_movie in user.get_predictions().keys():
            user.user_based_CF(predicting_movie, training_data, 20, "Pearson")
        user.output_predictions(output_file)
    output_file.close()


def run_test(userid, movieid):
    """
    Run a test on one testing user to predict one movie's rating
    """
    training_data = read_train(TRAINING_URL, TRAINING_USERS, TRAINING_MOVIES)
    user = read_test(TESTING5_URL)[userid]
    print "user's rated movies:", user.get_rated_movies()

    user.user_based_CF(movieid, training_data, 5, "Pearson")
    print "prediction", user.get_predictions()


# run_test(201, 1)

run_example()

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

# l = np.array([[1,2,3,0], [4,5,6,7], [8,9,10,11],[12,13,14,15]])
# # print l
# l2 = np.ma.masked_equal(l, 0)
# ave1 = np.ma.average(l, 1)
# ave2 = np.ma.average(l2,1)
# print ave1
# print ave2
# ave_t = ave.reshape((len(ave), 1))
# print np.hstack((l, ave_t))
# print np.average(l, 1)
# l2 = np.array([1,2,3,0])
# print np.argsort(l2)
# print np.argsort(l2)[::-1][0:2]
# print l[:, -1][np.argsort(l2)[::-1][0:2]]
# d = np.vstack((l[:, -1], l2[0:3]))
# print d[0] * d[1]
# print l
# print l - 1
# print list(l -2)
# non_zero_ids = np.nonzero(l2)
# print non_zero_ids, type(non_zero_ids), type(non_zero_ids[0])
# print l[:,non_zero_ids[0]]
# predict_non_zero = np.nonzero(l[:, 3])[0]
# print "non_zero_row", predict_non_zero
# movie_ids = np.array([0,2])
# print "movie ids", movie_ids
# b = np.array([predict_non_zero, movie_ids])
# c = np.array([[1,2], [0,2]])
# print l[b]
# print l[c]
# y = np.arange(35).reshape(5,7)
# d = 3
# a = np.array([0,1, d])
# print y
# print y[a,:][:, np.array([0,1])]
# d = 3
# a = np.array([0,1])
# c = np.hstack((a, d))
# print c[0: -1]
# d = {1:0, 3:4, 2:10}
# rate = []
# for key in sorted(d.keys()):
#     rate.append(d[key])
# print rate
# l2 = np.array([1,2,3,0, -1])
# b = l2 > 0
# print b, sum(b), type(b)
# print l2[b]
# c = l2 < 0
# print c
# l3 = l2.astype('float')
# print l2 * l2

# def cosine_similarity(vect2, vect1):
#             """
#             Compute the basic cosine similarity between vect1 and vect2 with the same dimensions.
#             vect1 can't contain zero value.
#             Vect1 and vect2 are np.arrays
#             """
#             assert sum(vect1 <= 0) == 0, "Test User's rating score has zero value"
#             non_zeros = vect2 > 0
#             if sum(non_zeros) == 0:
#                 # vector2 has no rated scores
#                 return 0.0
#             if sum(non_zeros) == 1:
#                 # fix the issue that if vectors have dimension of 1, their cosine similarity is always 1
#                 cosine = 0.8 - float(abs(vect1[non_zeros] - vect2[non_zeros])) * 0.2
#             else:
#                 vector1 = vect1[non_zeros].astype('float')
#                 vector2 = vect2[non_zeros].astype('float')
#                 cosine = sum(vector1 * vector2) / np.sqrt(sum(vector1 * vector1) * sum(vector2 * vector2))
#             return cosine
# #
# print cosine_similarity(np.array([0,0,0]), np.array([1,2,3]))
# print cosine_similarity(np.array([0,0,3]), np.array([1,2,3]))
# print cosine_similarity(np.array([0,0,1]), np.array([1,2,3]))
# print cosine_similarity(np.array([1,2,0]), np.array([1,2,3]))
# print cosine_similarity(np.array([3,2,1]), np.array([1,2,3]))
# print cosine_similarity(np.array([1,2,3]), np.array([1,2,3]))
#
# l = np.array([[0,0,0], [0,0,3], [0,0,1], [1,2,0], [3,2,1], [1,2,3]])


# print np.apply_along_axis(cosine_similarity, 1, l, np.array([1,2,3]))
# user_rated_scores = np.array([3, 2, -1, 4, -5.3, 4.3])
# if np.size(np.nonzero(user_rated_scores - np.average(user_rated_scores))) == 0:
#     print "T", np.size(np.nonzero(user_rated_scores - np.average(user_rated_scores)))
#
# print int(np.average(user_rated_scores))
# print user_rated_scores - np.average(user_rated_scores)
# print np.absolute(user_rated_scores)



