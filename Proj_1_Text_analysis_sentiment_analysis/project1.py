from string import punctuation, digits
import numpy as np
import random

# Part I


#pragma: coderesponse template
def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices
#pragma: coderesponse end


#pragma: coderesponse template
def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """

    dot_product = np.dot(theta, feature_vector.transpose())
    z = (dot_product + theta_0) * label
    if z >= 1:
        hinge_loss = 0
    else:
        hinge_loss = 1-z
    return (hinge_loss)
    # Your code here
    raise NotImplementedError
#pragma: coderesponse end


#pragma: coderesponse template
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code here
    thetas_test = np.reshape(theta, [1, -1])
    theta_add = np.reshape(theta, [1, -1])
    for i in range(len(feature_matrix) - 1):
        thetas_test = np.concatenate([thetas_test, theta_add], axis=0)

    theta_0s_test = np.reshape(theta_0, [1, -1])
    theta_0_add = np.reshape(theta_0, [1, -1])
    for i in range(len(feature_matrix) - 1):
        theta_0s_test = np.concatenate([theta_0s_test, theta_0_add], axis=0)

    test22 = map(hinge_loss_single, feature_matrix, labels, thetas_test, theta_0s_test)
    te = np.asarray(list(test22))
    return (np.mean(te))

    raise NotImplementedError
#pragma: coderesponse end


#pragma: coderesponse template
def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    testing_f = ((np.dot(current_theta, feature_vector.transpose()))+current_theta_0) * label

    if abs(testing_f - 0) <= np.finfo(np.float).eps:
        testing_f = 0
    else:
        pass
    if testing_f <= 0:
        return(current_theta + label * feature_vector, current_theta_0+label)
    else:
        return(current_theta, current_theta_0)
    raise NotImplementedError
#pragma: coderesponse end


#pragma: coderesponse template
def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    theta_int = np.zeros(feature_matrix[0].shape)
    theta_0_int = np.asarray([0])
    result = (theta_int,theta_0_int)
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            result=perceptron_single_step_update(feature_matrix[i], labels[i], result[0], result[1])
            pass
    return(result)
    raise NotImplementedError
#pragma: coderesponse end


#pragma: coderesponse template
def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Your code here
    theta_int = np.zeros(feature_matrix[0].shape)
    theta_0_int = np.asarray([0])
    result = [theta_int, theta_0_int]
    suma = [theta_int, theta_0_int]
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            result = perceptron_single_step_update(feature_matrix[i], labels[i], result[0], result[1])
            suma[0] = suma[0] + result[0]
            suma[1] = suma[1] + result[1]
            pass
    suma[0] = suma[0] / (len(feature_matrix) * T)
    suma[1] = suma[1] / (len(feature_matrix) * T)
    return (tuple(suma))
    raise NotImplementedError
#pragma: coderesponse end


#pragma: coderesponse template
def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    testing_f = ((np.dot(current_theta, feature_vector.transpose())) + current_theta_0) * label

    if abs(testing_f - 1) <= np.finfo(np.float).eps:
        testing_f = 1
    else:
        pass
    if testing_f <= 1:
        return ((1 - eta * L) * current_theta + eta * label * feature_vector, current_theta_0 + eta * label)
    else:
        return ((1 - eta * L) * current_theta, current_theta_0)
    raise NotImplementedError
#pragma: coderesponse end


#pragma: coderesponse template
def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # Your code here
    theta_int = np.zeros(feature_matrix[0].shape)
    theta_0_int = np.asarray([0])
    result = (theta_int, theta_0_int)
    j = 1
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            eta_int = 1 / np.sqrt(j)
            j = j + 1
            result = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta_int, result[0], result[1])
            pass
    return (result)
    raise NotImplementedError
#pragma: coderesponse end

# Part II


#pragma: coderesponse template
def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Your code here
    result_mat = np.zeros((len(feature_matrix)))
    countt=0
    for i in feature_matrix:
        testing_f = ((np.dot(theta, i.transpose())) + theta_0)
        if abs(testing_f - 0) <= np.finfo(np.float).eps:
            testing_f = 0
        else:
            pass
        if testing_f > 0:
            result_mat[countt] = 1
        else:
            result_mat[countt] = -1
        countt=countt+1
    return result_mat
    raise NotImplementedError
#pragma: coderesponse end


#pragma: coderesponse template
def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Your code here
    theta_res, theta_0_res = classifier(train_feature_matrix, train_labels, **kwargs)

    prediction_train = classify(train_feature_matrix, theta_res, theta_0_res)
    accu_train = accuracy(prediction_train, train_labels)

    prediction_val = classify(val_feature_matrix, theta_res, theta_0_res)
    accu_val = accuracy(prediction_val, val_labels)

    return (accu_train, accu_val)
    raise NotImplementedError
#pragma: coderesponse end


#pragma: coderesponse template
def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()
#pragma: coderesponse end


#pragma: coderesponse template
def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # Your code here
    dictionary = []  # maps word to unique index
    dictionary_dic = {}
    dict = open("stopwords.txt").read()
    test = extract_words(dict)
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary.append(word)
    for i in test:
        for j in dictionary:
            if i == j:
                del dictionary[dictionary.index(j)]
    for word in dictionary:
        if word not in dictionary_dic:
            dictionary_dic[word] = len(dictionary_dic)

    return dictionary_dic


#pragma: coderesponse end



#pragma: coderesponse template
def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = word_list.count(word)
    return feature_matrix
#pragma: coderesponse end


#pragma: coderesponse template
def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
#pragma: coderesponse end
