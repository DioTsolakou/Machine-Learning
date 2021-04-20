import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# globals for now
w1 = []  # Mx(D+1) matrix, j line has vector wj
w2 = []  # Kx(M+1) matrix, k line has vector wk
x = []  # input data of (D+1)-dimensions
pick = 0


def stochastic_gradient_ascent():
    print("Choose activation function.\n 0 for log, 1 for exp, 2 for cos.\n")
    global pick
    pick = input()
    #cost(w, K, N, lamda)  # to be determined later


def gradcheck():   # function to check partial derivative
    pass


def cost(w, X, t, lamda):
    cost_of_w = 0
    N, D = X.shape
    K = t.shape[1]

    for n in range(N):
        for k in range(K):
            cost_of_w += (t[n][k] * np.log(y(K, k, n))) - (lamda/2) * np.power(w, 2)  # t to be declared later

    return cost_of_w


def y(K, index_k, index_n):
    denominator = 0
    for i in range(K):
        denominator += np.exp(np.transpose(w2[i]) * z(index_n, 0))  # j is 0 for now

    return np.exp(np.transpose(w2[index_k]) * z(index_n, 0)) / denominator  # j is 0 for now


def z(n, j):
    return activation_func(np.transpose(w1[j]) * x[n], )


def activation_func(a):
    if pick == 0:
        return np.log(1 + np.exp(a))
    elif pick == 1:
        return (np.exp(a) - np.exp(-a)) / np.exp(a) + np.exp(-a)
    else:
        return np.cos(a)


def weight_init(f_in, f_out):
    return np.random.uniform(-np.sqrt(6/(f_in + f_out)), np.sqrt(6/(f_in + f_out)))


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
        print("Now loading file : data/mnist/{0}{1}.txt".format(file, i))
        tmp = pd.read_csv("data/mnist/{0}{1}.txt".format(file, i), header=None, sep=" ")
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


def view_dataset(train_data):
    n = 100
    sqrt_n = int(n**0.5)
    samples = np.random.randint(train_data.shape[0], size=n)

    plt.figure(figsize=(11, 11))

    counter = 0
    for i in samples:
        counter += 1
        plt.subplot(sqrt_n, sqrt_n, counter)
        plt.subplot(sqrt_n, sqrt_n, counter).axis('off')
        plt.imshow(train_data[i].reshape(28, 28), cmap='gray')

    plt.show()


def main():
    train_data, test_data, y_train, y_test = load_data()
    view_dataset(train_data)
    #stochastic_gradient_ascent()


if __name__ == "__main__":
    main()
