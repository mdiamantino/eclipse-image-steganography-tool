from math import exp
from random import randrange

import numpy as np


# ============================================================

class POBA_GA:
    def __init__(self, P_c, P_m, alpha, T, N):
        self.maxZ_A_t0 = None
        self.T = T
        self.N = N
        self.alpha = alpha
        self.P_c, self.P_m = P_c, P_m
        self.fs = np.zeros((T, N))
        self.frs = np.zeros((T, N))
        self.phis = np.zeros((T, N))
        self.gamma = 0 # TODO

    @staticmethod
    def generateRandomNoise(rows, cols, mean, var):
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (rows, cols))
        return gaussian

    def main(self):
        """

        :param P_c:  Probability of crossover
        :param P_m: Probability of mutation
        :param alpha: Proportional coefficient used to adjust the proportion of attack performance and perturbation size.
        :param T: Maximum number of iterations
        :param N: Population size
        :return:
        """
        y_0 = ""  # The true label of the original example S
        y_1 = ""  # The output label
        gamma = 0  # todo
        S = np.array()  # original image TODO
        random_noises_A = []
        for i in range(self.N):
            random_noises_A.append(self.generateRandomNoise(rows=S.shape[0],
                                                            cols=S.shape[1],
                                                            mean=0,
                                                            var=1))
        A__t = np.array(random_noises_A)
        self.maxZ_A_t0 = max(self.Z(perturbation) for perturbation in A__t)
        for t in range(self.T):
            AS__t = A__t + S  # Collection of the t_th iteration adversarial examples
            for i in range(self.N):
                y_0 = ""  # True label of the adversarial image
                y_1 = ""  # Label with highest confidence of the target method for AS_i_t (output label)
                y_2 = ""  # Label with second highest confidence of the target method for AS_i_t
                if y_1 != y_0:
                    fit_value = self.confidence(y_1, AS__t[i]) - self.confidence(y_0, AS__t[i]) - (
                            (self.alpha / self.maxZ_A_t0) * self.Z(A__t[i]))
                else:
                    fit_value = self.confidence(y_2, AS__t[i]) - self.confidence(y_0, AS__t[i])
                self.phis[t][i] = fit_value
            for i in range(self.N):
                self.fs[t][i] = self.phis[t][i] / sum(self.phis[t][j] for j in range(self.N))
                self.frs[t][i] = sum(self.fs[t][j] for j in range(i))
            ##
            # if max...
            ##
            for n in range(self.N / 2):
                A__t[i] = self.selection(AS__t)
                A__t[j] = self.selection(AS__t)
                A_2n_1_t_plus_1, A_2n_t_plus_1 = self.crossover(A__t[i], A__t[j], self.P_c)
                A_2n_1_t_plus_1 = self.mutation(A_2n_1_t_plus_1, self.P_m)
                A_2n_t_plus_1 = self.mutation(A_2n_t_plus_1, self.P_m)

    def Z(self, A_i_t):
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

    def confidence(self, y, AS_i_t):
        """
        Calculates the confidence that the adversarial example AS_i_t is labeled as y by the target method
        :return:
        """
        return

    def phi(self, AS_i_t):
        """
        For a given AS_i_t, calculates its fitness function.
        :param AS_i_t: The tth iteration of the ith adversarial example
        :return:
        """
        y_0 = ""  # True label of the adversarial image
        y_1 = ""  # Label with highest confidence of the target method for AS_i_t (output label)
        y_2 = ""  # Label with second highest confidence of the target method for AS_i_t
        A_i_t = 0  # TODO
        if y_1 != y_0:
            return self.confidence(y_1, AS_i_t) - self.confidence(y_0, AS_i_t) - (
                    (self.alpha / self.maxZ_A_t0) * self.Z(A_i_t))
        else:
            return self.confidence(y_2, AS_i_t) - self.confidence(y_0, AS_i_t)

    def f(self, AS_i_t, list_of_AS_i_t):
        """
        For a given AS_i_t, calculates its selection probability.
        :param AS_i_t: The tth iteration of the ith adversarial example
        :return: Selection probability
        """
        n = 0  # TODO
        return self.phi(AS_i_t) / sum(self.phi(list_of_AS_i_t[j] for j in range(n)))

    def fr(self, AS_i_t, list_of_AS_i_t):
        """
        The cumulative probability of AS_i_t
        to produce two children by crossover and mutation operations
        :return:
        """
        i = 0  # TODO (index of AS_i_t)
        return sum(self.f(list_of_AS_i_t[j]) for j in range(i))

    def selection(self, AS__t):
        """
        Let choose two parent examples to produce two children by crossover and mutation operators.
        :param AS__t: The collection of t_th iteration adversarial examples
        (when t=0, it represents the initial adversarial example)
        :return:
        """
        i = 0
        list_of_AS_i_t = []  # TODO
        while 0 <= self.fr(AS__t[i], list_of_AS_i_t) <= 1:
            i += 1
        return AS__t[i]

    # ============================================================

    def crossover(self, A_i_t, A_j_t, P_c=1):
        """
        Executed to generate new examples from selected examples A_i_t, A_j_t
        :param P_c: Probability of crossover (the larger, the faster the faster the fitness function converges)
        :param A_i_t: Parent perturbation
        :param A_j_t: Parent perturbation
        :return: new examples
        """
        # A__t[i], A__t[j] are two parent perturbations
        height, width = 0, 0  # TODO
        B = np.random.randint(2, size=(height, width))  # two-dimensional matrix in crossover
        if randrange(0, 1, 0.01) < P_c:
            b_opposite = (1 - B)
            A_i_t_plus_1 = A_i_t * B + A_j_t * b_opposite
            A_j_t_plus_1 = A_i_t * b_opposite + A_j_t * B
        else:
            A_i_t_plus_1 = A_i_t
            A_j_t_plus_1 = A_j_t
        return A_i_t_plus_1, A_j_t_plus_1

    def mutation(self, A_i_t_plus_1, P_m=0.001):
        """
        Alters some example by a certain probability during the breeding process.+
        :param P_m: Probability of mutation
        :param A_i_t_plus_1:
        :return:
        """
        height, width = 0, 0  # TODO
        C = np.random.randint(3, size=(height, width))  # two-dimensional matrix in crossover
        if randrange(0, 1, 0.01) < P_m:
            return A_i_t_plus_1 * C
        else:
            return A_i_t_plus_1


"""
def P(AS_i_t):
    # Calculates the attack performance of example AS_i_t
    # :param AS_i_t: The tth iteration of the ith adversarial example
    # :return: Attack performance of example AS_i_t
    y_0 = ""  # True label of the adversarial image
    y_1 = ""  # Label with highest confidence of the target method for AS_i_t (output label)
    y_2 = ""  # Label with second highest confidence of the target method for AS_i_t
    if y_1 != y_0:
        return confidence(y_1, AS_i_t) - confidence(y_0, AS_i_t)
    else:
        return confidence(y_2, AS_i_t) - confidence(y_0, AS_i_t)
# """
