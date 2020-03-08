from random import random

import cv2
import efficientnet.keras as efn
import numpy as np
from keras.applications.inception_v3 import preprocess_input, decode_predictions


# ============================================================

class POBA_GA:
    def __init__(self, image_path, T, N, alpha=3, P_c=1, P_m=0.001):
        # self.app = ClarifaiApp(api_key='7c87e299629e4e7ea0566aca3136b214')
        # self.model = self.app.public_models.general_model
        self.model = efn.EfficientNetB0(weights='imagenet')  # or weights='noisy-student'
        self.S = cv2.imread(image_path)
        self.height, self.width = self.S.shape[0], self.S.shape[1]
        self.maxZ_A_t0 = None
        self.T = T
        self.N = N
        self.alpha = alpha
        self.P_c, self.P_m = P_c, P_m
        self.fs = np.zeros((N))
        self.frs = np.zeros((N))
        self.phis = np.zeros((N))
        self.gamma = 0.1  # TODO
        self.A, self.AS = None, None
        self.y_0, self.p_0, _, _ = self.getLables(self.S)  # True label of the adversarial image
        # self.y_0, self.p_0 = "tree", 0.92
        # Parameters to adjust the perturbation pixel mapping rule (if naked eye, pm1=10, pm2=5.8)
        self.pm1, self.pm2 = 10, 5.8
        self.mean, self.variance = 0, 0.5
        self.initialization()
        print("ok")

    def initialization(self):
        random_noises_A = []
        for i in range(self.N):
            random_noises_A.append(
                self.generateRandomNoise(self.S.shape[:2], mean=self.mean, var=self.variance))
        self.A = np.array(random_noises_A)
        self.A.resize((2, self.N, *self.S.shape[:2]), refcheck=False)  # Add one dimension
        self.AS = np.zeros((self.N, *self.S.shape[:2], 3))
        self.maxZ_A_t0 = max(self.Z(i) for i in range(len(self.A[0])))
        # print(self.maxZ_A_t0)
        # self.mergeImgPlusNoise(t=0)


    def getLables(self, np_img):
        """success, encoded_image = cv2.imencode('.png', np_img)
        if not success:
            raise TypeError
        response = self.model.predict_by_bytes(encoded_image.tobytes(), is_video=False)
        concepts = response['outputs'][0]['data']['concepts']
        # Label with highest confidence of the target method for AS_i_t (output label)
        y_1, p_1 = concepts[0]['name'], concepts[0]['value']
        # Label with second highest confidence of the target method for AS_i_t
        y_2, p_2 = concepts[1]['name'], concepts[1]['value']
        """
        # print("-> PREDICTING")
        x = cv2.resize(np_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.model.predict(x)
        top2 = decode_predictions(preds, top=2)[0]
        p = [{'label': description, 'probability': probability}
             for label, description, probability in top2]
        y_1, p_1, y_2, p_2 = p[0]['label'], p[0]['probability'], p[1]['label'], p[1]['probability']
        if p_1 < 0.5:
            print(p_1)
        return y_1, p_1, y_2, p_2

    @staticmethod
    def generateRandomNoise(shape, mean, var):
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, shape)
        return gaussian


    def mergeImgPlusNoise(self):
        for channel_index in range(3):
            self.AS[:, :, :, channel_index] = self.S[:, :, channel_index] + self.A[0, :, :, :]

    def main(self):
        """
        :param P_c:  Probability of crossover
        :param P_m: Probability of mutation
        :param alpha: Proportional coefficient used to adjust the proportion of attack performance and perturbation size.
        :param T: Maximum number of iterations
        :param N: Population size
        :return:
        """
        optimal_t = 0
        for t in range(self.T):
            print("Starting iteration n°", t)
            # Collection of the t_th iteration adversarial examples
            self.mergeImgPlusNoise()
            for i in range(self.N):
                self.phis[i] = self.phi(i)
            for i in range(self.N):
                self.fs[i] = self.f(i)
                self.frs[i] = self.fr(i)
            if max(self.phis) > self.gamma:
                optimal_t = t
                break
            for n in range(self.N // 2):
                i = self.selection()  # First chosen parent
                j = self.selection()  # Second chosen parent
                # assert i != j
                A_2n_1_t_plus_1, A_2n_t_plus_1 = self.crossover(self.A[0][i],
                                                                self.A[0][j])
                self.A[1][2 * n - 1] = self.mutation(A_2n_1_t_plus_1)
                self.A[1][2 * n] = self.mutation(A_2n_t_plus_1)
            self.A[0] = self.A[1]
            self.A[1] = np.zeros(self.A.shape[1:])
            self.A.resize((t + 2, self.N, *self.S.shape[:2]), refcheck=False)
        optimal_img_indx = np.argmax(self.phis)
        optimal_res = self.AS[optimal_img_indx]
        print(optimal_res)
        return optimal_res

    def Z(self, i):
        """
        Calculates the perturbation size of adversarial example.
        It is a perturbation calculation indicator.
        :param A_i_t:  The tth iteration of the ith ith perturbation
        :return: Perturbation size
        """
        var = (-np.abs(self.AS[i]) * self.pm1 + self.pm2)
        first_exp = np.exp(var)
        second_exp = np.exp(self.pm2)
        perturbation_size = (1 / (1 + first_exp) - 1 / (1 + second_exp))
        return np.sum(perturbation_size)

    def phi(self, i):
        """
        For a given AS_i_t, calculates its fitness function.
        :param AS_i_t: The tth iteration of the ith adversarial example
        :return:
        """
        current_perturbation_img = self.A[0][i]
        current_adversarial_img = self.AS[i]
        y_1, p_1, y_2, p_2 = self.getLables(current_adversarial_img)
        if y_1 != self.y_0:
            print("----> Different LABELS")
            print("DIFFERENT LABELS : %s != %s" % (y_1, y_2))
            return p_1 - self.p_0 - (
                    (self.alpha / self.maxZ_A_t0) * self.Z(i))
        else:
            return p_2 - self.p_0

    def f(self, i):
        """
        For a given AS_i_t, calculates its selection probability.
        :param AS_i_t: The tth iteration of the ith adversarial example
        :return: Selection probability
        """
        return self.phis[i] / sum(self.phis[j] for j in range(self.N))

    def fr(self,i):
        """
        The cumulative probability of AS_i_t
        to produce two children by crossover and mutation operations
        :return:
        """
        return sum(self.fs[j] for j in range(i))

    def selection(self):
        """
        Let choose two parent examples to produce two children by crossover and mutation operators.
        :param AS__t: The collection of t_th iteration adversarial examples
        (when t=0, it represents the initial adversarial example)
        :return:
        """
        i = 0
        while i < self.N and random() < self.frs[i]:
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
        if random() < self.P_c:
            # print("ok")
            b_opposite = (1 - B)
            A_i_t_plus_1 = A_i_t * B + A_j_t * b_opposite
            A_j_t_plus_1 = A_i_t * b_opposite + A_j_t * B
        else:
            # print("not okay")
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
        if random() < self.P_m:
            return A_i_t_plus_1 * C
        else:
            return A_i_t_plus_1


if __name__ == "__main__":
    poba_ga = POBA_GA("data/test_image.jpg", 5000, 20)
    poba_ga.main()
