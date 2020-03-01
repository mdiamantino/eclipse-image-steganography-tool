from math import exp


def Z(A_i_t):
    """
    Calculates the perturbation size of adversarial example.
    It is a perturbation calculation indicator.
    :param A_i_t:  The tth iteration of the ith ith perturbation
    :return: Perturbation size
    """
    m_a = A_i_t.shape[0]  # Height of the image
    m_b = A_i_t.shape[1]  # Width of the image
    pm1, pm2 = 10, 5.8  # Parameters to adjust the perturbation pixel mapping rule (if naked eye, pm1=10, pm2=5.8)
    perturbation_size = 0
    for a in range(m_a):
        for b in range(m_b):
            perturbation_size += (1 / (1 + exp((-abs(A_i_t[a][b]) * pm1 + pm2)))) - (1 / (1 + exp(pm2)))
    return perturbation_size


def confidence(y, AS_i_t):
    """
    Calculates the confidence that the adversarial example AS_i_t is labeled as y by the target method
    :return:
    """
    return


def P(AS_i_t):
    """
    Calculates the attack performance of example AS_i_t
    :param AS_i_t: The tth iteration of the ith adversarial example
    :return: Attack performance of example AS_i_t
    """
    y_0 = ""  # True label of the adversarial image
    y_1 = ""  # Label with highest confidence of the target method for AS_i_t (output label)
    y_2 = ""  # Label with second highest confidence of the target method for AS_i_t
    if y_1 != y_0:
        return confidence(y_1, AS_i_t) - confidence(y_0, AS_i_t)
    else:
        return confidence(y_2, AS_i_t) - confidence(y_0, AS_i_t)


def phi(AS_i_t):
    """
    For a given AS_i_t, calculates its fitness function.
    :param AS_i_t: The tth iteration of the ith adversarial example
    :return:
    """
    y_0 = ""  # True label of the adversarial image
    y_1 = ""  # Label with highest confidence of the target method for AS_i_t (output label)
    y_2 = ""  # Label with second highest confidence of the target method for AS_i_t
    alpha = 0  # Proportional coefficient used to adjust the proportion of attack performance and perturbation size.
    maxZ_A_t0 = 0  # TODO
    A_i_t = 0  # TODO
    if y_1 != y_0:
        return confidence(y_1, AS_i_t) - confidence(y_0, AS_i_t) - ((alpha / maxZ_A_t0) * Z(A_i_t))
    else:
        return confidence(y_2, AS_i_t) - confidence(y_0, AS_i_t)


def f(AS_i_t, list_of_AS_i_t):
    """
    For a given AS_i_t, calculates its selection probability.
    :param AS_i_t: The tth iteration of the ith adversarial example
    :return: Selection probability
    """
    n = 0  # TODO
    return phi(AS_i_t) / sum(phi(list_of_AS_i_t[j] for j in range(n)))


def selection(AS_i_t, ):
    """
    The cumulative probability of AS_i_t
    to produce two children by crossover and mutation operations
    :return:
    """
    i = 0  # TODO (index of AS_i_t)
    resrun
