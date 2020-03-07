import numpy as np


def generatePerturbation(shape, mean, var):
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, shape)
    return gaussian


def applyToPixel(pixel):
    pm1, pm2 = 10, 5.8
    var = (-abs(pixel) * pm1 + pm2)
    first_exp = np.exp(var)
    second_exp = np.exp(pm2)
    perturbation_size = (1 / (1 + first_exp) - 1 / (1 + second_exp))
    return perturbation_size


def Z(A_i_t):
    """
    Calculates the perturbation size of adversarial example.
    It is a perturbation calculation indicator.
    :param A_i_t:  The tth iteration of the ith ith perturbation
    :return: Perturbation size
    """
    vf = np.vectorize(applyToPixel)
    new = vf(A_i_t)
    print(new.shape)
    return np.sum(new)


def Zplus(A_i_t):
    pm1, pm2 = 10, 5.8
    var = (-np.abs(A_i_t) * pm1 + pm2)
    first_exp = np.exp(var)
    second_exp = np.exp(pm2)
    perturbation_size = (1 / (1 + first_exp) - 1 / (1 + second_exp))
    return np.sum(perturbation_size)


if __name__ == "__main__":
    noise = generatePerturbation(shape=(1920, 1080), mean=0, var=1)
    print(Zplus(noise))
