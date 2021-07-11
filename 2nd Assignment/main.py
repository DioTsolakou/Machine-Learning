import random
import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy.stats
import datetime
from dateutil.relativedelta import relativedelta


def init_params(X, k):
    probabilities = np.full(shape=k, fill_value=1/k)  # is Kx1
    probabilities = np.array(probabilities)
    means = np.asarray(random.sample(list(X), k))  # is Kx3
    means = np.array(means)
    sigma = [np.cov(X.T) for _ in range(k)]  # is of length K
    sigma = np.array(sigma)
    return probabilities, means, sigma


def load_img(file):
    original_img = imageio.imread(file)
    imgplot = plt.imshow(original_img)
    plt.xlabel("Original image")
    plt.show()
    #print("original shape : ", original_img.shape)
    #print("original content : ", original_img)
    return original_img


def flatten_img(file):
    x, y, z = file.shape
    img_2d = file.reshape(x * y, z)
    img_2d = np.array(img_2d, dtype=float)
    print("2d shape : ", img_2d.shape)
    return img_2d


def from_2d_to_3d(file, x, y, k):
    file = np.array(file, dtype=np.uint8)
    #file = (file * 255).astype(np.uint8)
    img_3d = file.reshape(x, y, 3)
    #print("reconstructed img shape : ", img_3d.shape)
    #print("reconstructed img content : ", img_3d)
    imgplot = plt.imshow(img_3d)
    plt.xlabel('k = {}'.format(k))
    plt.show()
    return img_3d


def get_GMM(x, pis, means, sigma, k):
    #sigma = np.array([np.eye(x.shape[1]) * s for s in sigma])
    gmm = np.array([pis[i] * scipy.stats.multivariate_normal.pdf(x, mean=means[i], cov=sigma[i]) for i in range(k)]).T
    return gmm  # is X[0]xK, 379500xK


def new_gamma(x, pis, means, sigma, k):
    gamma = get_GMM(x, pis, means, sigma, k)
    denom = np.sum(gamma, axis=1)
    denom = np.reshape(denom, (len(denom), 1))
    gamma = gamma / denom
    return gamma  # is X[0]xK, 379500xK


def new_pis(gamma):
    pis = np.sum(gamma, axis=0) / gamma.shape[0]  # sum of gamma / size of gamma (N)
    #pis = np.mean(gamma, axis=0)
    pis = np.array(pis)
    return pis  # is Kx1


def new_means(x, gamma):
    means = []
    for i in range(gamma.shape[1]):
        gamma_sum_div = gamma[:, i] / np.sum(gamma[:, i])
        gamma_sum_div = np.reshape(gamma_sum_div, (1, len(gamma_sum_div)))
        elements = np.dot(gamma_sum_div, x)
        means.append(elements[0])
    means = np.array(means)

    #means = np.sum(np.dot(gamma, x)) / np.sum(gamma, axis=0)
    return means  # is Kx3


def new_sigmas(x, gamma, means):
    sigma = []
    for i in range(gamma.shape[1]):
        weight = gamma[:, i] / np.sum(gamma[:, i])
        weight = np.reshape(weight, (1, len(weight)))
        sigmas = [np.mat(j - means[i]).T * np.mat(j - means[i]) for j in x]
        cov_j = sum(weight[0][i] * sigmas[i] for i in range(len(weight[0])))
        sigma.append(cov_j)
    sigma = np.array(sigma)
    #print("reached new sigma", sigma)
    return sigma  # needs to be Kx3x3


def test_new_sigmas(x, gamma):
    sigmas = []
    for i in range(gamma.shape[1]):
        gamma_w = gamma[:, [i]]
        gamma_sum = np.sum(gamma_w)
        sigma = np.cov(x.T, aweights=(np.divide(gamma_w, gamma_sum)).flatten(), bias=True)
        sigmas.append(sigma)

    sigmas = np.array(sigmas)
    #print("test new sigma shape ", sigmas.shape)
    return sigmas  # is Kx3x3


def improved_new_sigma(x, gamma, means):
    tmp_list = []
    for i in range(gamma.shape[1]):
        tmp = x - means[i]
        tmp_list.append(tmp)
    tmp_list = np.array(tmp_list)

    enumerator = np.sum(np.sum(np.sum(np.dot(gamma.T, np.square(tmp_list)), axis=0), axis=1))
    denominator = np.sum(np.dot(means.shape[1], np.sum(gamma, axis=0)))  # must be Nx3
    sigma_value = enumerator / denominator

    sigma = []
    for i in range(gamma.shape[1]):
        tmp = np.array(np.identity(means.shape[1]) * sigma_value)
        sigma.append(tmp)

    sigma = np.array(sigma)
    return sigma  # is Kx3x3


def new_log_likelihood(x, pis, means, sigma, k):
    gmm = get_GMM(x, pis, means, sigma, k).T
    likelihood = np.log(np.sum(gmm, axis=0))
    log_sum = np.sum(likelihood)
    return log_sum


def step_E(x, pis, means, sigma, k):
    gamma = new_gamma(x, pis, means, sigma, k)
    return gamma


def step_M(x, gamma, k):
    pis = new_pis(gamma)
    means = new_means(x, gamma)
    #sigmas = new_sigmas(x, gamma, means)
    sigmas = improved_new_sigma(x, gamma, means)
    #sigmas = test_new_sigmas(x, gamma)
    return pis, means, sigmas


def max_pis(pis):
    max_pi = np.argmax(pis, axis=1)
    return max_pi


def error_func(original_img, reconstructed_img):
    if original_img.shape != reconstructed_img.shape:
        return -1

    original_img_norm_np = np.linalg.norm(original_img, axis=-1)
    reconstructed_img_norm_np = np.linalg.norm(reconstructed_img, axis=-1)

    original_img_norm = np.sum(np.square(original_img), axis=0, keepdims=True)
    original_img_norm = np.sum(original_img_norm, axis=1, keepdims=True)
    original_img_norm = np.sqrt(original_img_norm)
    original_img_norm = np.array(original_img_norm)

    reconstructed_img_norm = np.sum(np.square(reconstructed_img), axis=0, keepdims=True)
    reconstructed_img_norm = np.sum(reconstructed_img_norm, axis=1, keepdims=True)
    reconstructed_img_norm = np.sqrt(reconstructed_img_norm)
    reconstructed_img_norm = np.array(reconstructed_img_norm)

    error1 = np.square(np.sum(original_img_norm) - np.sum(reconstructed_img_norm)) / np.multiply(original_img.shape[0], original_img.shape[1])
    error2 = np.sum(np.square(original_img_norm_np - reconstructed_img_norm_np) / np.multiply(original_img.shape[0], original_img.shape[1]))

    print("Error1 is : ", error1)
    print("Error2 is : ", error2)


def ml_train(x, pis, means, sigma, iterations, k):
    tol = 1e-5
    iter = 0
    likelihood = 0
    new_likelihood = 2

    for i in range(iterations):
        start_dt = datetime.datetime.now()
        likelihood = new_likelihood

        # E step
        gamma = step_E(x, pis, means, sigma, k)
        # M step
        pis, means, sigmas = step_M(x, gamma, k)
        # Calculate log likelihood
        new_likelihood = new_log_likelihood(x, pis, means, sigma, k)
        max_pi = max_pis(gamma)
        end_dt = datetime.datetime.now()
        diff = relativedelta(end_dt, start_dt)
        print("iter: %s, time interval: %s:%s:%s:%s" % (i, diff.hours, diff.minutes, diff.seconds, diff.microseconds))
        print("log-likelihood = {}".format(new_likelihood))

        if np.abs(likelihood - new_likelihood) < tol:
            break

    return max_pi, pis, means, sigma


def main():
    k = int(input("Give the desired k : "))
    iterations = int(input("Give the desired amount of iterations : "))
    original_img = load_img("im.jpg")
    img_2d = flatten_img(original_img)
    pis, means, sigma = init_params(img_2d, k)
    max_pi, pis, means, sigma = ml_train(img_2d, pis, means, sigma, iterations, k)
    new_image = means[max_pi]
    reconstructed_img = from_2d_to_3d(new_image, original_img.shape[0], original_img.shape[1], k)
    error_func(original_img, reconstructed_img)

    print("original img content ", original_img)
    print("reconstructed img content ", reconstructed_img)


if __name__ == "__main__":
    main()
