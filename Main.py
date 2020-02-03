
import numpy
#remove types

from EditDistance import distance_by_errors
import pickle

dictionary = pickle.load(open("dictionary_5000.pickle", "rb"))
data = pickle.load(open("data_50000.pickle", "rb"))


errors_vector = numpy.zeros(4)
for i in range(0,len(data)-5000):
    temp_errors = distance_by_errors(data[i][1],data[i][0])
    errors_vector = numpy.add(temp_errors,errors_vector)


distribution_vector = [ 14706,  24100,   2743,  33638]
reversed_distribution_vector = []
vectors_sum = sum(distribution_vector)
for i in range(0,4):
    distribution_vector[i] = distribution_vector[i]/vectors_sum
    reversed_distribution_vector.append(1-distribution_vector[i])
"""
print(distribution_vector)
print(reversed_distribution_vector)
"""
def return_distance(weight_vector, distance):
    actual_distance = 0
    for i in range(4):
        actual_distance += weight_vector[i]*distance[i]
    return actual_distance


def learn(weight_vector, word, correct_word):
    min3_words = [0,0,0]
    dict_corel = numpy.zeros(len(dictionary))
    for j in range(len(dictionary)):
        dict_corel[j] = return_distance(weight_vector, distance_by_errors(dictionary[j][1],word))
    first_min = numpy.argmin(dict_corel)
    min3_words[0] = dictionary[first_min][1]
    dict_corel[first_min] = 400
    second_min = numpy.argmin(dict_corel)
    min3_words[1] = dictionary[second_min][1]
    dict_corel[second_min] = 400
    third_min = numpy.argmin(dict_corel)
    min3_words[2] = dictionary[third_min][1]
    if correct_word in min3_words:
        return 0
    else:
        return 1


def error():
    error_counter = 0
    for i in range(45000,45100):
        error_counter += learn(reversed_distribution_vector, data[i][1], data[i][0])
    return error_counter/100



print(error())

