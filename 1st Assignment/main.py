import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# globals for now
w1 = []  # Mx(D+1) matrix, j line has vector wj
w2 = []  # Kx(M+1) matrix, k line has vector wk
x = []  # input data of (D+1)-dimensions
pick = 0


def stochastic_gradient_ascent(train_data, t):
    print("Choose activation function.\n 0 for log, 1 for exp, 2 for cos.\n")
    global pick
    pick = input()
    print("Choose hidden unit size")
    hidden_size = input()

    w = weight_norm(train_data, hidden_size, t)


def weight_norm(X, hidden, t):
    w1_size = (hidden, X.shape+1)
    w2_size = (t.shape[1], hidden+1)
    w1 = weight_init(X.shape, hidden, w1_size)
    w2 = weight_init(X.shape, hidden, w2_size)

    w1 = np.sum(np.square(w1), axis=0), np.sum(np.square(w1), axis=1)
    w2 = np.sum(np.square(w2), axis=0), np.sum(np.square(w2), axis=1)
    w = np.sum(w1, w2)

    return w


def cost(w, X, t, lamda):
    cost_of_w = 0
    N, D = X.shape
    K = t.shape[1]

    y = softmax(np.dot(X, w.T))

    for n in range(N):
        for k in range(K):
            cost_of_w += (t[n][k] * np.log(y)) - (lamda/2) * np.power(w, 2)

    gradEw = np.dot((t - y).T, X) - lamda * w

    return cost_of_w, gradEw


def softmax(__x, ax=1):
    m = np.max(__x, axis=ax, keepdims=True)  # max per row
    p = np.exp(__x - m)
    return p / np.sum(p, axis=ax, keepdims=True)


def z(n, j):
    return activation_func(np.transpose(w1[j]) * x[n])


def activation_func(a):
    if pick == 0:
        return np.log(1 + np.exp(a))
    elif pick == 1:
        return (np.exp(a) - np.exp(-a)) / np.exp(a) + np.exp(-a)
    else:
        return np.cos(a)


def weight_init(f_in, f_out, shape):
    return np.random.uniform(-np.sqrt(6/(f_in + f_out)), np.sqrt(6/(f_in + f_out)), shape)


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


def ml_softmax_train(t, X, lamda, w, options):
    max_iter = options[0]
    tol = options[1]
    lr = options[2]

    Ewold = -np.inf
    costs = []

    for i in range(1, max_iter+1):
        E, gradEw = cost(w, X, t, lamda)
        costs.append(E)
        if i % 50 == 0:
            print('Iteration : %d, Cost function :%f' % (i, E))

        if np.abs(E - Ewold) < tol:
            break

        w = w + lr * gradEw
        Ewold = E

    return w, costs


def gradcheck_softmax(Winit, X, t, lamda):
    W = np.random.rand(*Winit.shape)
    epsilon = 1e-6

    _list = np.random.randint(X.shape[0], size=5)
    x_sample = np.array(X[_list, :])
    t_sample = np.array(t[_list, :])

    E, gradEw = cost(W, x_sample, t_sample, lamda)

    print("gradEw shape: ", gradEw.shape)

    numericalGrad = np.zeros(gradEw.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    for k in range(numericalGrad.shape[0]):
        for d in range(numericalGrad.shape[1]):
            # add epsilon to the w[k,d]
            w_tmp = np.copy(W)
            w_tmp[k, d] += epsilon
            e_plus, _ = cost(w_tmp, x_sample, t_sample, lamda)

            # subtract epsilon to the w[k,d]
            w_tmp = np.copy(W)
            w_tmp[k, d] -= epsilon
            e_minus, _ = cost(w_tmp, x_sample, t_sample, lamda)

            # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)

    return gradEw, numericalGrad


def grad_difference(X, t):
    N, D = X.shape
    K = t.shape[1]
    Winit = np.zeros((K, D))
    lamda = 0.1
    options = [500, 1e-6, 0.5/N]

    gradEw, numericalGrad = gradcheck_softmax(Winit, X, t, lamda)

    print("The difference estimate for the gradient of w is : ", np.max(np.abs(gradEw - numericalGrad)))


def training(X, t):
    N, D = X.shape
    K = t.shape[1]
    Winit = np.zeros((K, D))
    lamda = 0.1
    options = [500, 1e-6, 0.5/N]

    w, costs = ml_softmax_train(t, X, lamda, Winit, options)
    return costs, options, w


def ml_softmax_test(W, X_test):
    ytest = softmax(X_test.dot(W.T))
    # Hard classification decisions
    ttest = np.argmax(ytest, 1)
    return ttest


def accuracy(W, X_test, y_test):
    pred = ml_softmax_test(W, X_test)
    print(np.mean(pred == np.argmax(y_test, 1)))


def plot_results(costs, options):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(format(options[2], 'f')))
    plt.show()


def main():
    train_data, test_data, y_train, y_test = load_data()
    view_dataset(train_data)
    stochastic_gradient_ascent(train_data, y_train)
    costs, options, w = training(train_data, y_train)
    plot_results(costs, options)
    accuracy(w, test_data, y_test)


if __name__ == "__main__":
    main()
