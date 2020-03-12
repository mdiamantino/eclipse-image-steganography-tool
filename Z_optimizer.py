import numpy as np
from skimage.util import random_noise

def generatePerturbation(shape, mean, var):
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, shape)
    return gaussian.astype('uint8')

def genPert(shape, mean, var):
    noise = random_noise(np.zeros(shape), mode="gaussian", mean=mean, var=var)
    noise_img = np.array(255 * noise, dtype='uint8')
    return noise_img

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
    normalized = A_i_t.astype('float64')/255
    pm1, pm2 = 10, 5.8
    var = (-np.abs(normalized) * pm1) + pm2
    first_exp = np.exp(var)
    perturbation_size =( (1 / (1 + first_exp)) - (1 / (1 + np.exp(pm2))))
    return np.sum(perturbation_size)/(A_i_t.shape[0]*A_i_t.shape[1])


if __name__ == "__main__":
    noise = generatePerturbation(shape=(1920, 1080, 3), mean=0, var=1)
    print(Zplus(noise))
    # pert = genPert(shape=(1920, 1080), mean=0, var=.2)
    # print(Zplus(pert))


