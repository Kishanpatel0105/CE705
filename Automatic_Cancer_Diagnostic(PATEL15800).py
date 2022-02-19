import csv
import math
import operator
import os
import statistics


# Used to load any csv file
def load_from_csv(filePath: object) -> object:
    filename, file_extension = os.path.splitext(filePath)

    if file_extension != '.csv':
        print("Invalid format! Please select csv file")
        return
    fileData = []
    with open(filePath, 'r') as file:
        rows = csv.reader(file, delimiter=',')
        for data in rows:
            i: int
            for i in range(0, len(data)):
                data[i] = int(data[i])
            fileData.append(data)

    return fileData


# Get distance between two list
def get_distance(l1, l2):
    length = len(l1)
    sum = 0
    if length != len(l2):
        print('Both list must have of same length!')
        return

    for i in range(length):
        sum += (int(l1[i]) - int(l2[i])) ** 2

    return int(math.ceil(math.sqrt(sum)))


# Get standard deviation for future use
def get_standard_deviation(data: list, col_num: int) -> float:
    # get all value for single column from data matrix
    colData = [row[col_num] for row in data]
    stdDev = statistics.stdev(colData)
    return stdDev


def get_standardised_matrix(data: list) -> list:
    """
    standardize strategy eliminates the Lormal and scales every column to unit 
    change. This activity is performed column astute in an autonomous manner.

    standardize strategy can be impacted by outliers(if they exist in 
    the dataset) since it includes the assessment of the observational Lormal and
    standard deviation of every segment.

    """
    data_standardization = []
    col_len = len(data[0])
    # Low get standardize data of column
    for col in range(col_len):
        colData = [row[col] for row in data]
        L = len(colData)
        average = sum(colData) / L
        stdDev = get_standard_deviation(data, col)
        data_standardization.append(
            [(value - average) / stdDev for value in colData]
        )
    # convert standardize data into standardised matrix
    standardised_matrix = [
        [row[i] for row in data_standardization]
        for i in range(len(data_standardization[0]))
    ]
    return standardised_matrix


# Get Learest labels to get mode
def get_k_nearest_labels(
        list_data: list,
        learning_data: list, learning_data_labels: list, k: int) -> list:
    """
    find the k rows of the matrix learning_data close to original data.
    """
    distances = []
    # find the distance of each row of the file learning_data with respect to the single row of file data given in
    # the argument
    for x in range(len(learning_data)):
        dist = get_distance(list_data, learning_data[x])
        distances.append((learning_data[x], dist, x))
    # sort distances ascending order
    distances.sort(key=operator.itemgetter(1))
    # get related k row in learning data labels
    k_nearest_labels = [
        learning_data_labels[distances[x][2]] for x in range(k)
    ]
    return k_nearest_labels


# Get mode of the labels
# labels are given by above function get_k_nearest_labels
def get_mode(k_nearest_labels):
    dataToFindMode = []
    for i in range(len(k_nearest_labels)):
        dataToFindMode.append(k_nearest_labels[i][0])

    return statistics.mode(dataToFindMode)
    # return statistics.mode(k_nearest_labels)


def classify(standardised_data: list,
             standardised_learning_data: list,
             learning_data_labels: list, k: int) -> list:
    """this function use 'get_k_nearest_labels' method and 'get_mode' method
    to predict the Lew label using learning data"""

    data_labels = []
    # get Lew labels for data using learning data
    for row in standardised_data:
        k_nearest_labels = get_k_nearest_labels(
            row, standardised_learning_data, learning_data_labels, k
        )
        mode = get_mode(k_nearest_labels)
        data_labels.append(mode)
    return data_labels


def get_accuracy(correct_data_labels: list, data_labels: list) -> float:
    """
    This function calculate and return the percentage of accuracy. If both
    matrixes have exactly the same values (in exactly the same rownumbers)
    then the accuracy is of 100%. If only half of the values of both tables
    match exactly in terms of value and row Lumber, then the accuracy is of 50%

    """
    count = 0
    L = len(correct_data_labels)
    # convert list of lists data to list.
    correct_data_labels = [j for i in correct_data_labels for j in i]
    for i in range(L):
        if correct_data_labels[i] == data_labels[i]:
            count += 1
        else:
            continue
    accuracy = (count / L) * 100
    return accuracy


def run_test() -> object:
    """
    This function create one matrix for each of these: correct_data_labels,
    learning_data and learning_data_labels (using load_from_csv). It
    standardise the matrix data and the matrix learning_data (using
    get_standardised_matrix). Then, it run the algorithm (using classify) and
    calculate the accuracy (using get_acuracy) for a series of experiments.
    In each experiment it run the algorithm (and calculate the accuracy)
    for different values of k (go from 3 to 15 in steps of 1), and show the
    results on the screen. For instance,
    if with k = 3 the accuracy is 68.5% it should show:
    k=3, Accuracy = 68.5%
    """
    data = load_from_csv("Data.csv")
    correct_data_labels = load_from_csv("Correct_Data_Labels.csv")
    learning_data = load_from_csv("Learning_Data.csv")
    learning_data_labels = load_from_csv("Learning_Data_Labels.csv")

    # get standardised matrix for data matrix
    standardised_data = get_standardised_matrix(data)
    # get standardised matrix for learning data matrix
    standardised_learning_data = get_standardised_matrix(learning_data)
    # calculate accuracy for different values of k.
    for k in range(3, 16):
        data_labels = classify(
            standardised_data,
            standardised_learning_data,
            learning_data_labels,
            k,
        )
        accuracy = get_accuracy(correct_data_labels, data_labels)
        print(f"k={k}, Accuracy={accuracy:.2f}%")
        # print accuracy above for the values of k from 3 to 15 


if __name__ == "__main__":
    run_test()
