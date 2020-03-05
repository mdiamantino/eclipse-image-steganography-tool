from math import exp
from random import randrange

import cv2
import numpy as np
from clarifai.rest import ClarifaiApp


# ============================================================

class POBA_GA:
    def __init__(self, image_path, T, N, alpha=3, P_c=1, P_m=0.001):
        self.app = ClarifaiApp(api_key='7c87e299629e4e7ea0566aca3136b214')
        self.model = self.app.public_models.general_model
        self.S = cv2.imread(image_path)
        self.height, self.width = self.S.shape[0], self.S.shape[1]
        self.maxZ_A_t0 = None
        self.T = T
        self.N = N
        self.alpha = alpha
        self.P_c, self.P_m = P_c, P_m
        self.fs = np.zeros((T, N))
        self.frs = np.zeros((T, N))
        self.phis = np.zeros((T, N))
        self.gamma = 0.1  # TODO
        self.A, self.AS = None, None
        self.y_0, self.p_0, _, _ = self.getLables(self.S)  # True label of the adversarial image
        # Parameters to adjust the perturbation pixel mapping rule (if naked eye, pm1=10, pm2=5.8)
        self.pm1, self.pm2 = 10, 5.8
        self.initialization()

    def getLables(self, np_img):
        success, encoded_image = cv2.imencode('.png', np_img)
        if not success:
            raise TypeError
        response = self.model.predict_by_bytes(encoded_image.tobytes(), is_video=False)
        concepts = response['outputs'][0]['data']['concepts']
        # Label with highest confidence of the target method for AS_i_t (output label)
        y_1, p_1 = concepts[0]['name'], concepts[0]['value']
        # Label with second highest confidence of the target method for AS_i_t
        y_2, p_2 = concepts[1]['name'], concepts[1]['value']
        return y_1, p_1, y_2, p_2

    @staticmethod
    def generateRandomNoise(shape, mean, var):
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, shape)
        return gaussian

    def initialization(self):
        random_noises_A = []
        for i in range(self.N):
            random_noises_A.append(self.generateRandomNoise(self.S.shape[:2], mean=0, var=1))
        self.A = np.array(random_noises_A)
        self.A.resize((2, *self.A.shape))  # Add one dimension
        self.AS = np.zeros(self.A.shape)
        self.maxZ_A_t0 = max(self.Z(perturbation) for perturbation in self.A[0])
        self.mergeImgPlusNoise(t=0)

    def mergeImgPlusNoise(self, t):
        for i in range(3):
            self.AS[t][:, :, i] = self.S[:, :, i] + self.A[t]

    def main(self):
        """
        :param P_c:  Probability of crossover
        :param P_m: Probability of mutation
        :param alpha: Proportional coefficient used to adjust the proportion of attack performance and perturbation size.
        :param T: Maximum number of iterations
        :param N: Population size
        :return:
        """
        last_t = 0
        for t in range(self.T):
            # Collection of the t_th iteration adversarial examples
            self.mergeImgPlusNoise(t)
            for i in range(self.N):
                self.phis[t][i] = self.phi(t, i)
            for i in range(self.N):
                self.fs[t][i] = self.f(t, i)
                self.frs[t][i] = self.fr(t, i)
            if max(self.phis[t]) > self.gamma:
                last_t = t
                break
            for n in range(self.N / 2):
                i = self.selection(t)  # First chosen parent
                j = self.selection(t)  # Second chosen parent
                assert i != j
                A_2n_1_t_plus_1, A_2n_t_plus_1 = self.crossover(self.A[t][i],
                                                                self.A[t][j])
                self.A[t + 1][2 * n - 1] = self.mutation(A_2n_1_t_plus_1)
                self.A[t + 1][2 * n] = self.mutation(A_2n_t_plus_1)
            new_shape = list(self.A.shape)
            new_shape[0] += 1
            self.A.resize(new_shape)
            self.AS.resize(new_shape)
        index_of_optimal = np.argmax(self.phis)
        optimal_res = self.AS[last_t][index_of_optimal]
        return optimal_res

    def Z(self, A_i_t):
        """
        Calculates the perturbation size of adversarial example.
        It is a perturbation calculation indicator.
        :param A_i_t:  The tth iteration of the ith ith perturbation
        :return: Perturbation size
        """
        m_a = A_i_t.shape[0]  # Height of the image
        m_b = A_i_t.shape[1]  # Width of the image
        perturbation_size = 0
        for a in range(m_a):
            for b in range(m_b):
                perturbation_size += (1 / (
                            1 + exp((-abs(A_i_t[a][b]) * self.pm1 + self.pm2)))) - (
                                             1 / (1 + exp(self.pm2)))
        return perturbation_size

    def phi(self, t, i):
        """
        For a given AS_i_t, calculates its fitness function.
        :param AS_i_t: The tth iteration of the ith adversarial example
        :return:
        """
        AS_i_t = self.AS[t][i]
        A_i_t = self.A[t][i]
        y_1, p_1, y_2, p_2 = self.getLables(AS_i_t)
        if y_1 != self.y_0:
            return p_1 - self.p_0 - (
                    (self.alpha / self.maxZ_A_t0) * self.Z(A_i_t))
        else:
            return p_2 - self.p_0

    def f(self, t, i):
        """
        For a given AS_i_t, calculates its selection probability.
        :param AS_i_t: The tth iteration of the ith adversarial example
        :return: Selection probability
        """
        return self.phi(t, i) / sum(self.phi(t, j) for j in range(self.N))

    def fr(self, t, i):
        """
        The cumulative probability of AS_i_t
        to produce two children by crossover and mutation operations
        :return:
        """
        return sum(self.f(t, j) for j in range(i))

    def selection(self, t):
        """
        Let choose two parent examples to produce two children by crossover and mutation operators.
        :param AS__t: The collection of t_th iteration adversarial examples
        (when t=0, it represents the initial adversarial example)
        :return:
        """
        i = 0
        while 0 <= self.fr(t, i) <= 1:
            i += 1
        return i

    # ============================================================

    def crossover(self, A_i_t, A_j_t):
        """
        Executed to generate new examples from selected examples A_i_t, A_j_t
        :param P_c: Probability of crossover (the larger, the faster the faster the fitness function converges)
        :param A_i_t: Parent perturbation
        :param A_j_t: Parent perturbation
        :return: new examples
        """
        # A__t[i], A__t[j] are two parent perturbations

        # two-dimensional matrix in crossover
        B = np.random.randint(2, size=(self.height, self.width))
        if randrange(0, 1, 0.01) < self.P_c:
            b_opposite = (1 - B)
            A_i_t_plus_1 = A_i_t * B + A_j_t * b_opposite
            A_j_t_plus_1 = A_i_t * b_opposite + A_j_t * B
        else:
            A_i_t_plus_1 = A_i_t
            A_j_t_plus_1 = A_j_t
        return A_i_t_plus_1, A_j_t_plus_1

    def mutation(self, A_i_t_plus_1):
        """
        Alters some example by a certain probability during the breeding process.+
        :param P_m: Probability of mutation
        :param A_i_t_plus_1:
        :return:
        """
        # two-dimensional matrix in crossover
        C = np.random.randint(3, size=(self.height, self.width))
        if randrange(0, 1, 0.01) < self.P_m:
            return A_i_t_plus_1 * C
        else:
            return A_i_t_plus_1


if __name__ == "__main__":
    poba_ga = POBA_GA("data/test_image.jpg", 300, 10)
