import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    w1 = weight_init(X.shape[1], hidden+1, w1_size)
    w1 = np.array(w1)
    global w2
    w2 = weight_init(X.shape[1], hidden+1, w2_size)
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

    w_norm = weight_norm(train_data, hidden_size, t)
    return w_norm, batch_size, hidden_size


def softmax(__x, ax=1):
    m = np.max(__x, axis=ax, keepdims=True)  # max per row
    p = np.exp(__x - m)
    return p / np.sum(p, axis=ax, keepdims=True)


def create_batches(X, t, batch_size):
    data_x = X
    np.random.shuffle(data_x)
    x_batches = []

    data_t = t
    np.random.shuffle(data_t)
    t_batches = []

    for i in range(int(data_x.shape[0] / batch_size)):
        batch_x = np.array(data_x[i * batch_size:(i+1) * batch_size, :])
        x_batches.append(batch_x)
        batch_t = np.array(data_t[i * batch_size:(i+1) * batch_size, :])
        t_batches.append(batch_t)

    return x_batches, t_batches


def cost_batch(w_norm, w1, w2, X, t, lamda, batch_size, hidden_size):
    cost_of_w = 0
    N = batch_size
    K = t.shape[1]
    counter = 0

    if counter == 0:
        x_batches, t_batches = create_batches(X, t, batch_size)
        counter = 1

    Z = np.array(shape=(N, hidden_size))

    for n in range(N):
        for j in range(hidden_size):
            Z[n][j] = activation_func(np.dot(x_batches[counter * n], w1[j].T))

    #Z = activation_func(np.dot(x_batches[counter], w1.T))
    y = softmax(np.dot(Z, w2[:, 1:].T))

    for n in range(N):
        for k in range(K):
            cost_of_w += (t_batches[counter * n][k] * np.log(y[counter * n][k]))
    cost_of_w -= (lamda/2) * np.power(w_norm, 2)

    Z_prime = np.array(shape=(N, hidden_size))

    for n in range(N):
        for j in range(hidden_size):
            Z_prime[n][j] = activation_func_prime(np.dot(x_batches[counter * n], w1[j].T))

    gradEw1 = np.dot((np.dot((t - y), w2[:, 1:]) * activation_func_prime(np.dot(X, w1.T))).T, X) - lamda * w1
    gradEw2 = np.dot((t - y).T, Z) - lamda * w2[:, 1:]

    counter += 1

    return cost_of_w, gradEw1, gradEw2


def cost(w_norm, w1, w2, X, t, lamda):
    cost_of_w = 0
    N = X.shape[0]
    K = t.shape[1]

    np.random.shuffle(X)

    Z = activation_func(np.dot(X, w1.T))  # (6000, 785) * (100, 785).T = (6000, 100)
    y = softmax(np.dot(Z, w2[:, 1:].T))

    max_error = np.max(np.dot(X, w1.T), axis=1)

    for n in range(N):
        for k in range(K):
            cost_of_w += (t[n][k] * np.log(y[n][k]))
    cost_of_w -= (lamda/2) * np.power(w_norm, 2)

#    cost_of_w = np.sum(t * y) - np.sum(max_error) - np.sum(np.log(np.sum(np.exp(y - np.array([max_error, ] * y.shape[1]).T), 1))) - (0.5 * lamda) * np.square(w_norm)

    gradEw1 = np.dot((np.dot((t - y), w2[:, 1:]) * activation_func_prime(np.dot(X, w1.T))).T, X) - lamda * w1
    gradEw2 = np.dot((t - y).T, Z) - lamda * w2[:, 1:]

    return cost_of_w, gradEw1, gradEw2


def activation_func(a):
    if pick == 0:
        return np.log(1 + np.exp(np.abs(a)))
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
    return np.random.uniform(-np.sqrt(6/(f_in + f_out)), np.sqrt(6/(f_in + f_out)), shape)


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


def ml_softmax_train(t, X, lamda, options, w1, w2, w_norm, batch_size, hidden_size):
    max_iter = options[0]
    tol = options[1]
    lr = options[2]

    Ewold = -np.inf
    costs = []

    for i in range(1, max_iter+1):
        E, gradEw1, gradEw2 = cost(w_norm, w1, w2, X, t, lamda)
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
            w_tmp_2 = np.copy(W2)
            w_tmp_1[k, d] += epsilon
            w_tmp_2[k, d] += epsilon
            cost_of_w, e_plus1, e_plus2 = cost(w_norm, w_tmp_1, w_tmp_2, x_sample, t_sample, lamda)

            # subtract epsilon to the w[k,d]
            w_tmp_1 = np.copy(W1)
            w_tmp_2 = np.copy(W2)
            w_tmp_1[k, d] -= epsilon
            w_tmp_2[k, d] -= epsilon
            cost_of_w, e_minus1, e_minus2 = cost(w_norm, w_tmp_1, w_tmp_2, x_sample, t_sample, lamda)

            # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGrad1[k, d] = (e_plus1 - e_minus1) / (2 * epsilon)
            numericalGrad2[k, d] = (e_plus2 - e_minus2) / (2 * epsilon)

    return gradEw1, gradEw2, numericalGrad1, numericalGrad2


def grad_difference(X, t, w1, w2, w_norm):
    N, D = X.shape
    K = t.shape[1]
    lamda = 0.1
    options = [500, 1e-6, 0.05]

    gradEw1, gradEw2, numericalGrad1, numericalGrad2 = gradcheck_softmax(w1, w2, X, t, lamda, w_norm)

    print("The difference estimate for the gradient of w1 is : ", np.max(np.abs(gradEw1 - numericalGrad1)))
    print("The difference estimate for the gradient of w2 is : ", np.max(np.abs(gradEw2 - numericalGrad2)))


def training(X, t, w1, w2, w_norm, batch_size, hidden_size):
    N, D = X.shape
    K = t.shape[1]
    lamda = 0.1
    options = [500, 1e-6, 0.005]

    w1, w2, costs = ml_softmax_train(t, X, lamda, options, w1, w2, w_norm, batch_size, hidden_size)
    return costs, options, w1, w2


def ml_softmax_test(W, X_test):
    ytest = softmax(X_test.dot(W.T))
    # Hard classification decisions
    ttest = np.argmax(ytest, 1)
    return ttest


def accuracy(X_test, y_test, X, t):
    N, D = X.shape
    K = t.shape[1]
    w_init = np.zeros((K, D))
    pred = ml_softmax_test(w_init, X_test)
    print("Accuracy : ", np.mean(pred == np.argmax(y_test, 1)))


def plot_results(costs, options):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = " + str(format(options[2], 'f')))
    plt.interactive(False)
    plt.show()


def main():
    train_data, test_data, y_train, y_test = load_data()
    view_dataset(train_data)

    train_data = train_data.astype(float)/255
    test_data = test_data.astype(float)/255

    train_data = np.hstack((np.ones((train_data.shape[0], 1)), train_data))
    test_data = np.hstack((np.ones((test_data.shape[0], 1)), test_data))

    w_norm, batch_size, hidden_size = initialize(train_data, y_train)
    batch_size = int(batch_size)
    hidden_size = int(hidden_size)
    global w1, w2
    costs, options, w1, w2 = training(train_data, y_train, w1, w2, w_norm, batch_size, hidden_size)
    plot_results(costs, options)
    accuracy(test_data, y_test, train_data, y_train)
    grad_difference(train_data, y_train, w1, w2, w_norm)


if __name__ == "__main__":
    main()
