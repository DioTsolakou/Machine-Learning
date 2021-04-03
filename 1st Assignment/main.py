import math
import numpy as np

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
