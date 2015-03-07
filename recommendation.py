__author__ = 'liuyincheng'

"""
Develop different algorithms to make recommendations for movies.
"""

import urllib2
import numpy as np
import math



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


def item_item_similarity(training_data, num_movie, method = "Adjusted_cosine", case_amplification = False):
    """
    :param training_data: movie rating matrix, row represents user and column represent movie. Both user id and movie id
    are indexed from 1.
    :return: item_item similarity nested dictionary
    """
    sim_matrix = {}
    if method == "Adjusted_cosine":
        # Calculate item-item similarity by Adjusted Cosine similarity
        masked_data = np.ma.masked_equal(training_data, 0)
        user_bias = np.ma.average(masked_data, 1)
        # user_bias = user_bias.reshape(len(user_bias), 1)
        for movie_i in range(1, num_movie + 1):
            sim_matrix[movie_i] = {}
            for movie_j in range(1, num_movie + 1):
                common_indices = training_data[:, movie_i - 1] * training_data[:, movie_j - 1] > 0
                if np.sum(common_indices) >= 5:
                    rating_i = training_data[common_indices, movie_i - 1] - user_bias[common_indices]
                    rating_j = training_data[common_indices, movie_j - 1] - user_bias[common_indices]
                    # assert np.sum(rating_i > 0) == np.sum(rating_j > 0), "vector i and vector j don't have same effective values \n" + str(training_data[common_indices, movie_i - 1]) + str(user_bias[common_indices, :])
                    similarity = np.sum(rating_i * rating_j) / np.sqrt(np.sum(rating_i * rating_i) * np.sum(rating_j * rating_j))
                else:
                    similarity = 0
                assert -1.0001 < similarity < 1.0001, "Similarity " + str(similarity) + " is out of range [-1, 1]"
                sim_matrix[movie_i][movie_j] = similarity

    elif method == "Cosine":
        # Calculate item-item similarity by Cosine similarity
        for movie_i in range(1, num_movie + 1):
            sim_matrix[movie_i] = {}
            rating_i = training_data[:, movie_i - 1]
            for movie_j in range(1, num_movie + 1):
                rating_j = training_data[:, movie_j - 1]
                common_indices = rating_i * rating_j > 0
                if np.sum(common_indices) >= 5:
                    common_rating_i = rating_i[common_indices]
                    common_rating_j = rating_j[common_indices]
                    assert len(common_rating_i) == len(common_rating_j), "vector i and vector j do not have same dimension"
                    similarity = np.sum(common_rating_i * common_rating_j) / np.sqrt(np.sum(common_rating_i * common_rating_i) * np.sum(common_rating_j * common_rating_j))
                else:
                    similarity = 0
                assert 0.0 <= similarity <= 1.0, "Similarity" + str(similarity) + " is out of range [0, 1]"

                if case_amplification == True:
                    similarity = math.pow(similarity, 5)

                sim_matrix[movie_i][movie_j] = similarity

    return sim_matrix


def average_rating(data, num_movie):
    """
    :param data: user-movie rating matrix
    :return: dictionary of movie id and it's average rating
    """
    masked_data = np.ma.masked_equal(data, 0)
    averages = np.ma.average(masked_data, 0)
    averages.filled(3)
    ave_dict = {}
    for movie in range(1, num_movie + 1):
        ave_dict[movie] = averages[movie - 1]
    return ave_dict


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


    def user_based_CF(self, predict_movie, training_data, num_nearest_neighbors, similarity_method, case_amplification = False, IFU = False, smooth = False):
        """
        :param predict_movie: the movie needed to predict
        :param training_data: training data (user-movie array matrix)
        :param num_nearest_neighbors: k
        :param similarity_method: Cosine similarity or Pearson correlation
        :param case_amplification: true if applied
        :param IFU: true if inverse user frequency applied
        :param smooth: true if linear smooth applied
        :return: predict_movie's prediction, range [1, 5]
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


        def pearson_correlation(vect2, vect1, case_ampl = case_amplification):
            """
            Calculate the pearson_correlation of vect1 and vect2 and
            return the weights.
            Vect1 represents the test user and can't have zero element(s).
            Last two values of vect2 are predicting movie's rating and neighbor's bias
            case_ampl: Apply case amplification or not.
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

            # Case amplification
            if case_ampl == True:
                case_ampl_para = 2.5
                weight = math.copysign(1, weight) * math.pow(abs(weight), case_ampl_para)
            assert -1.0001 < weight < 1.0001, "Weight " + str(weight) +" is out of range\n" +"training: " + str(vect2) + " Test user: " + str(vect1) + "\nvectors " + str(vector2) + str(vector1) + "\n average" + str(vect2_ave) + str(vect1_ave)

            return weight


        # Get relevant neighbors as a n * (m +1) matrix, where n is the number of neighbors,
        # first m columns are relevant movies and last column is predicting movie's score.
        rated_movies = self.get_rated_movies()
        rated_movies_indices = np.array(sorted(rated_movies.keys())) - 1
        predict_movie_index = predict_movie - 1
        predict_non_zero_id = np.nonzero(training_data[:, predict_movie_index])[0]

        # Get test user's rating as a np.array
        user_rated_scores = []
        for key in sorted(rated_movies.keys()):
            user_rated_scores.append(rated_movies[key])
        user_rated_scores = np.array(user_rated_scores)

        training_data_ifu = training_data

        if IFU == True:
            # Count inverse user frequency
            user_freq = np.clip(np.sum(training_data > 0.0, 0), 1.0, TRAINING_USERS)
            inverse_user_freq = np.log2(float(TRAINING_USERS) / user_freq.astype("float"))
            if inverse_user_freq[predict_movie_index] != 0:
                training_data_ifu = training_data * inverse_user_freq
                user_rated_scores = user_rated_scores * inverse_user_freq[rated_movies_indices]

        relevant_neighbors = training_data_ifu[predict_non_zero_id , :][:, np.hstack((rated_movies_indices, predict_movie_index))]

        # If the user doesn't have any preference on any movies, or this movie has not been rated by any other users,
        # Set prediction as the average of user's rating.
        if np.size(np.nonzero(user_rated_scores - np.average(user_rated_scores))) == 0 or np.size(relevant_neighbors) == 0:
            prediction = int(np.average(self.get_rated_movies().values()))
            assert 1 <= prediction <= 5, "Prediction is wrong" + str(prediction)
            self.update_prediction(predict_movie, prediction)
            return

        # Add bias to last column
        masked_neighbors = np.ma.masked_equal(training_data_ifu[predict_non_zero_id, :], 0)
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

            assert 1 <= prediction <= 5, "Prediction is wrong" + str(prediction)
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
                init_prediction = np.average(self.get_rated_movies().values())
            else:

                # print "weights", weights
                # print "k weights", weights[k_sorted_indices]
                if IFU == True:
                    # Get original users' rating and bias.
                    orig_user_average = np.average(self.get_rated_movies().values())
                    orig_relevant_neighbors = training_data[predict_non_zero_id , :][:, np.hstack((rated_movies_indices, predict_movie_index))]
                    orig_masked_neighbors = np.ma.masked_equal(training_data[predict_non_zero_id, :], 0)
                    orig_neighbor_bias = np.ma.average(orig_masked_neighbors, 1)
                    orig_relevant_neighbors = np.hstack((orig_relevant_neighbors, orig_neighbor_bias.reshape((len(orig_neighbor_bias), 1))))
                    orig_neighbor_natural_ratings = orig_relevant_neighbors[:,-2] - orig_relevant_neighbors[:, -1]
                    init_prediction = orig_user_average + sum(weights[k_sorted_indices] * orig_neighbor_natural_ratings[k_sorted_indices]) / sum(np.absolute(weights[k_sorted_indices]))
                else:
                    neighbor_natural_ratings = relevant_neighbors[:,-2] - relevant_neighbors[:, -1]
                    init_prediction = user_average + sum(weights[k_sorted_indices] * neighbor_natural_ratings[k_sorted_indices]) / sum(np.absolute(weights[k_sorted_indices]))

                # print sum(weights[k_sorted_indices] * neighbor_natural_ratings[k_sorted_indices]) / sum(np.absolute(weights[k_sorted_indices]))

            assert -3.000 < init_prediction < 20.000, "Prediction " + str(init_prediction) +" is out of reasonable range (0, 5]"

            if init_prediction < 1:
                prediction = 1
            elif init_prediction > 5:
                prediction = 5
            else:
                if smooth == True:
                    averages = average_rating(training_data, 1000)
                    init_prediction = 0.6 *init_prediction + 0.4 * averages[predict_movie]
                prediction = int(round(init_prediction))

            assert 1 <= prediction <= 5, "Prediction is wrong" + str(prediction)

            self.update_prediction(predict_movie, prediction)

        else:
            print "Wrong similarity method input"
            return None


    def item_based_CF(self, predict_movie, similarity_matrix):
        """
        Predict by item based algorithm, given a item_item_similarity_matrix
        """
        numerator = 0
        denominator = 0
        for movie, rating in self.get_rated_movies().items():
            similarity_score = similarity_matrix[predict_movie][movie]
            numerator += float(similarity_score * rating)
            denominator += float(abs(similarity_score))
        if denominator == 0 or numerator == 0:
            init_prediction = round(np.average(self.get_rated_movies().values()))
        else:
            init_prediction = numerator / denominator

        if init_prediction < 1:
            prediction = 1
        else:
            prediction = int(round(init_prediction))

        # print 'initial prediction', init_prediction
        assert 1 <= prediction <= 5, "Prediction is wrong" + str(prediction)
        self.update_prediction(predict_movie, prediction)


def run_user_based_example():
    """
    Run the recommendation system
    """
    training_data = read_train(TRAINING_URL, TRAINING_USERS, TRAINING_MOVIES)
    test_users = read_test(TESTING20_URL)
    output_file = open("result20_ub_pearson_10nbs_smooth_iuf.txt", "w")
    for user in test_users.values():
        for predicting_movie in user.get_predictions().keys():
            user.user_based_CF(predicting_movie, training_data, 10, "Pearson", case_amplification=False, IFU=True, smooth=True)
        user.output_predictions(output_file)
    output_file.close()


def run_user_based_test(userid, movieid):
    """
    Run a test on one testing user to predict one movie's rating
    """
    training_data = read_train(TRAINING_URL, TRAINING_USERS, TRAINING_MOVIES)
    user = read_test(TESTING5_URL)[userid]
    print "user's rated movies:", user.get_rated_movies()

    user.user_based_CF(movieid, training_data, 5, "Pearson", case_amplification=True, IFU=True)
    print "prediction of movie " + str(movieid), user.get_predictions()[movieid]


def run_item_based_example():
    """
    Predict rating by user based CF
    """
    training = read_train(TRAINING_URL, TRAINING_USERS, TRAINING_MOVIES)
    test_users = read_test(TESTING20_URL)
    similarity_matrix = item_item_similarity(training, TRAINING_MOVIES, method="Cosine", case_amplification=False)
    output_file = open("result20_ib_cosine8.txt", "w")
    for user in test_users.values():
        for predicting_movie in user.get_predictions().keys():
            user.item_based_CF(predicting_movie, similarity_matrix)
        user.output_predictions(output_file)
    output_file.close()


def test_suit1():
    training = np.array([[3,5,0,0,4,0,4,0],
                         [2,0,0,1,3,3,0,2],
                         [4,0,5,3,2,0,0,4],
                         [0,2,3,0,5,4,0,1],
                         [4,0,3,2,0,1,0,1],
                         [0,0,1,0,2,0,3,0],
                         [1,0,0,0,3,0,2,0],
                         [5,0,4,0,4,0,5,0],
                         [0,0,0,0,0,0,3,4]])
    test_1 = TestUser(101)
    test_1.add_rated_movie(1,3)
    test_1.add_rated_movie(2,5)
    test_1.add_rated_movie(3,2)
    test_1.add_rated_movie(4,2)
    test_1.add_rated_movie(5,4)
    test_1.add_rated_movie(6,4)
    test_1.add_prediction(7)
    test_1.add_prediction(8)
    similarity_matrix = item_item_similarity(training, 8, method="Cosine", case_amplification=True)
    for item in similarity_matrix:
        print str(item) + ": ", similarity_matrix[item]
    test_1.item_based_CF(7, similarity_matrix)
    print "prediction of movie 7", test_1.get_predictions()[7]
    print average_rating(training, 8)



run_user_based_example()

# run_item_based_example()

# run_user_based_test(229, 967)

# test_suit1()

