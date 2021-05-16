import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from six.moves import cPickle as pickle

# globals for now
w1 = []  # Mx(D+1) matrix, j line has vector wj
w2 = []  # Kx(M+1) matrix, k line has vector wk
x = []  # input data of (D+1)-dimensions
pick = 0


def weight_norm(X, hidden, t):
    hidden = int(hidden)
    w1_size = (hidden, X.shape[1])
    w2_size = (t.shape[1], hidden + 1)

    global w1
    w1 = weight_init(X.shape[1], hidden + 1, w1_size)
    w1 = np.array(w1)
    global w2
    w2 = weight_init(X.shape[1], hidden + 1, w2_size)
    w2 = np.array(w2)

    w1_n = w1
    w2_n = w2

    w1_n = np.sum(np.square(w1_n), axis=0, keepdims=True)
    w1_n = np.sum(w1_n, axis=1, keepdims=True)
    w2_n = np.sum(np.square(w2_n), axis=0, keepdims=True)
    w2_n = np.sum(w2_n, axis=1, keepdims=True)

    w1_n = np.array(w1_n)
    w2_n = np.array(w2_n)

    w_norm = w1_n + w2_n

    return w_norm


def initialize(train_data, t):
    print("Choose activation function.\n 0 for log, 1 for exp, 2 for cos.\n")
    global pick
    pick = input()
    print("Choose hidden unit size")
    hidden_size = input()
    print("Choose batch size")
    batch_size = input()
    print("Choose regularization value (lamda)")
    lamda = input()

    w_norm = weight_norm(train_data, hidden_size, t)
    return w_norm, batch_size, hidden_size, lamda


def softmax(__x, ax=1):
    m = np.max(__x, axis=ax, keepdims=True)  # max per row
    p = np.exp(__x - m)
    return p / np.sum(p, axis=ax, keepdims=True)


def cost(w_norm, w1, w2, X, t, lamda):
    cost_of_w = 0
    N = X.shape[0]
    K = t.shape[1]

    Z = activation_func(np.dot(X, w1.T))  # (6000, 785) * (100, 785).T = (6000, 100)
    Z = np.hstack((np.ones((Z.shape[0], 1)), Z))
    y = softmax(np.dot(Z, w2.T))

    max_error = np.max(np.dot(X, w1.T), axis=1)

    #    for n in range(N):
    #        for k in range(K):
    #            cost_of_w += (t[n][k] * np.log(y[n][k]))
    #    cost_of_w -= (lamda/2) * np.power(w_norm, 2)

    cost_of_w = np.sum(t * y) - np.sum(max_error) - np.sum(
        np.log(np.sum(np.exp(y - np.array([max_error, ] * y.shape[1]).T), 1))) - (0.5 * lamda) * np.square(w_norm)

    gradEw1 = np.dot((np.dot((t - y), w2[:, 1:]) * activation_func_prime(np.dot(X, w1.T))).T, X) - lamda * w1
    gradEw2 = np.dot((t - y).T, Z[:, 1:]) - lamda * w2[:, 1:]

    gradEw1 = np.array(gradEw1)
    gradEw2 = np.array(gradEw2)

    return cost_of_w, gradEw1, gradEw2


def activation_func(a):
    if pick == 0:
        # return np.log(1 + np.exp(np.abs(a)))
        return np.log(1 + np.exp(-np.abs(a))) + np.maximum(a, 0)
    elif pick == 1:
        return (np.exp(a) - np.exp(-a)) / np.exp(a) + np.exp(-a)
    else:
        return np.cos(a)


def activation_func_prime(a):
    if pick == 0:
        return np.exp(a) / (1 + np.exp(a))
    elif pick == 1:
        return 1 - np.square(np.tanh(a))
    else:
        return -np.sin(a)


def weight_init(f_in, f_out, shape):
    return np.random.uniform(-np.sqrt(6 / (f_in + f_out)), np.sqrt(6 / (f_in + f_out)), shape)


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


def view_dataset(train_data):
    n = 100
    sqrt_n = int(n ** 0.5)
    samples = np.random.randint(train_data.shape[0], size=n)

    plt.figure(figsize=(11, 11))

    counter = 0
    for i in samples:
        counter += 1
        plt.subplot(sqrt_n, sqrt_n, counter)
        plt.subplot(sqrt_n, sqrt_n, counter).axis('off')
        plt.imshow(train_data[i].reshape(28, 28), cmap='gray')

    plt.show()


def get_batch(X, t, batch_size):
    index = np.random.randint(len(X), size=batch_size)
    return X[index, :], t[index, :]


def ml_softmax_train(t, X, lamda, options, w1, w2, w_norm, batch_size):
    max_iter = options[0]
    tol = options[1]
    lr = options[2]

    Ewold = -np.inf
    costs = []

    for i in range(1, max_iter + 1):
        np.random.shuffle(X)
        np.random.shuffle(t)
        x_batch, t_batch = get_batch(X, t, batch_size)

        E, gradEw1, gradEw2 = cost(w_norm, w1, w2, x_batch, t_batch, lamda)

        costs.append(E)
        if i % 10 == 0:
            print('Iteration : %d, Cost function :%f' % (i, E))

        if np.abs(E - Ewold) < tol:
            break

        w1 = w1 + lr * gradEw1
        w2[:, 1:] = w2[:, 1:] + lr * gradEw2
        Ewold = E

    return w1, w2, costs


def gradcheck_softmax(w1, w2, X, t, lamda, w_norm):
    W1 = np.random.rand(*w1.shape)
    W2 = np.random.rand(*w2.shape)
    epsilon = 1e-6

    _list = np.random.randint(X.shape[0], size=5)
    x_sample = np.array(X[_list, :])
    t_sample = np.array(t[_list, :])

    E, gradEw1, gradEw2 = cost(w_norm, w1, w2, x_sample, t_sample, lamda)

    print("gradEw shape: ", gradEw1.shape)
    print("gradEw shape: ", gradEw2.shape)

    numericalGrad1 = np.zeros(gradEw1.shape)
    numericalGrad2 = np.zeros(gradEw2.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    for k in range(numericalGrad1.shape[0]):
        for d in range(numericalGrad1.shape[1]):
            # add epsilon to the w[k,d]
            w_tmp_1 = np.copy(W1)
            w_tmp_1[k, d] += epsilon
            e_plus1, _, _ = cost(w_norm, w_tmp_1, w2, x_sample, t_sample, lamda)

            # subtract epsilon to the w[k,d]
            w_tmp_1 = np.copy(W1)
            w_tmp_1[k, d] -= epsilon
            e_minus1, _, _ = cost(w_norm, w_tmp_1, w2, x_sample, t_sample, lamda)

            # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGrad1[k, d] = (e_plus1 - e_minus1) / (2 * epsilon)

    for i in range(numericalGrad2.shape[0]):
        for j in range(numericalGrad2.shape[1]):
            w_tmp_2 = np.copy(W2)
            w_tmp_2[i, j] += epsilon
            e_plus2, _, _ = cost(w_norm, w1, w_tmp_2, x_sample, t_sample, lamda)

            w_tmp_2 = np.copy(W2)
            w_tmp_2[i, j] -= epsilon
            e_minus2, _, _ = cost(w_norm, w1, w_tmp_2, x_sample, t_sample, lamda)

            numericalGrad2[i, j] = (e_plus2 - e_minus2) / (2 * epsilon)

    return gradEw1, gradEw2, numericalGrad1, numericalGrad2


def grad_difference(X, t, w1, w2, w_norm, lamda):
    gradEw1, gradEw2, numericalGrad1, numericalGrad2 = gradcheck_softmax(w1, w2, X, t, lamda, w_norm)
    print("The difference estimate for the gradient of w1 is : ", np.max(np.abs(gradEw1 - numericalGrad1)))
    print("The difference estimate for the gradient of w2 is : ", np.max(np.abs(gradEw2 - numericalGrad2)))


def training(X, t, w1, w2, w_norm, batch_size, lamda):
    options = [100, 1e-6, 0.05]
    w1, w2, costs = ml_softmax_train(t, X, lamda, options, w1, w2, w_norm, batch_size)
    return costs, options, w1, w2


def ml_softmax_test(w1, w2, X_test):
    z_test = activation_func(np.dot(X_test, w1.T))
    z_test = np.hstack((np.ones((z_test.shape[0], 1)), z_test))
    ytest = softmax(np.dot(z_test, w2.T))
    # Hard classification decisions
    ttest = np.argmax(ytest, 1)
    return ttest


def accuracy(X_test, y_test, w1, w2):
    pred = ml_softmax_test(w1, w2, X_test)
    print("Accuracy : ", np.mean(pred == np.argmax(y_test, 1)))


def plot_results(costs, options):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate = " + str(format(options[2], 'f')))
    plt.interactive(False)
    plt.show()


def start_NN(train_data, y_train, test_data, y_test):
    w_norm, batch_size, hidden_size, lamda = initialize(train_data, y_train)
    batch_size = int(batch_size)
    hidden_size = int(hidden_size)
    lamda = float(lamda)
    global w1, w2
    costs, options, w1, w2 = training(train_data, y_train, w1, w2, w_norm, batch_size, lamda)
    plot_results(costs, options)
    accuracy(test_data, y_test, w1, w2)
    grad_difference(train_data, y_train, w1, w2, w_norm, lamda)


def mnist():
    train_data, test_data, y_train, y_test = load_data()
    view_dataset(train_data)

    train_data = train_data.astype(float) / 255
    test_data = test_data.astype(float) / 255

    train_data = np.hstack((np.ones((train_data.shape[0], 1)), train_data))
    test_data = np.hstack((np.ones((test_data.shape[0], 1)), test_data))

    start_NN(train_data, y_train, test_data, y_test)


def load_pickle(f):
    return pickle.load(f, encoding='latin1')


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=6000, num_validation=1000, num_test=1000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'data/cifar10/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train, y_train, X_val, y_val, x_test, y_test


def cifar():
    train_data, y_train, x_val, y_val, test_data, y_test = get_CIFAR10_data()

    train_data = np.hstack((np.ones((train_data.shape[0], 1)), train_data))
    test_data = np.hstack((np.ones((test_data.shape[0], 1)), test_data))

    start_NN(train_data, y_train, test_data, y_test)


def main():
    print("Choose dataset to train on")
    print("0 for mnist, 1 for cifar")
    chosen = input()
    chosen = int(chosen)

    if chosen == 0:
        mnist()
    else:
        cifar()


if __name__ == "__main__":
    main()
