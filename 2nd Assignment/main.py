import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio


def init_params(X, k):
    #probabilities =
    means = np.asarray(random.sample(list(X), k))
    sigma = [np.cov(X.T) * np.identity(X.shape[1]) for _ in range(k)]
    print("means shape : ", means.shape)
    #print("means content : ", means)
    print("sigma shape : ", len(sigma))
    #print("sigma content : ", sigma)


def load_img(file):
    original_img = imageio.imread(file)
    imgplot = plt.imshow(original_img)
    plt.show()
    print("original shape : ", original_img.shape)
    #print("original content : ", original_img)
    return original_img


def flatten_img(file):
    x, y, z = file.shape
    img_2d = file.reshape(x * y, z)
    img_2d = np.array(img_2d, dtype=float)
    print("2d shape : ", img_2d.shape)
    return img_2d


def from_2d_to_3d(file, x, y):
    file = (file * 255).astype(np.uint8)
    img_3d = file.reshape(x, y, 3)
    print("reconstructed img shape : ", img_3d.shape)
    #print("reconstructed img content : ", img_3d)
    imgplot = plt.imshow(img_3d)
    plt.show()
    return img_3d


def error_func():
    return


def gaussian_distr():
    return


def ml_train():
    return


def main():
    k = int(input("Give the desired k : "))
    original_img = load_img("im.jpg")
    img_2d = flatten_img(original_img)
    init_params(img_2d, k)

    #from_2d_to_3d(img_2d, original_img.shape[0], original_img.shape[1])


if __name__ == "__main__":
    main()
