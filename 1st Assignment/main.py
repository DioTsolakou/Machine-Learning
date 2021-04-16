import math
import numpy as np
import pandas as pd

# globals for now
w1 = []  # Mx(D+1) matrix, j line has vector wj
w2 = []  # Kx(M+1) matrix, k line has vector wk
x = []  # input data of (D+1)-dimensions
pick = 0


def main():
    stochastic_gradient_ascent()


if __name__ == "__main__":
    main()


def stochastic_gradient_ascent():
    print("Choose activation function.\n 0 for log, 1 for exp, 2 for cos.\n")
    global pick
    pick = input()
    #cost(w, K, N, lamda)  # to be determined later


def gradcheck():   # function to check partial derivative
    pass


def cost(w, K, N, lamda):
    cost_of_w = 0

    for n in range(N):
        for k in range(K):
            cost_of_w += (t * math.log10(y(K, k, n))) - (lamda/2) * math.pow(np.linalg.norm(w), 2)  # t to be declared later

    return cost_of_w


def y(K, index_k, index_n):
    denominator = 0
    for i in range(K):
        denominator += math.exp(np.transpose(w2[i]) * z(index_n, 0))  # j is 0 for now

    return math.exp(np.transpose(w2[index_k]) * z(index_n, 0)) / denominator  # j is 0 for now


def z(n, j):
    return activation_func(np.transpose(w1[j]) * x[n], )


def activation_func(a):
    if pick == 0:
        return math.log10(1 + math.exp(a))
    elif pick == 1:
        return (math.exp(a) - math.exp(-a)) / math.exp(a) + math.exp(-a)
    else:
        return math.cos(a)


def load_data():
    """
    Load the MNIST dataset. Reads the training and testing files and create matrices.
    :Expected return:
    train_data:the matrix with the training data
    test_data: the matrix with the data that will be used for testing
    y_train: the matrix consisting of one
                        hot vectors on each row(ground truth for training)
    y_test: the matrix consisting of one
                        hot vectors on each row(ground truth for testing)
    """

    # load the train files
    train_data, y_train = load_file("train")

    # load test files
    test_data, y_test = load_file("test")

    return train_data, test_data, y_train, y_test


def load_file(file):
    array = []
    df = None
    
    for i in range(10):
        tmp = pd.read_csv('data/mnist/%s%d.txt' % file % i, header=None, sep=" ")
        # build labels - one hot vector
        hot_vector = [1 if j == i else 0 for j in range(0, 10)]

        for j in range(tmp.shape[0]):
            array.append(hot_vector)
        # concatenate dataframes by rows
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    data = df.to_numpy()
    array = np.array(array)
    
    return data, array
