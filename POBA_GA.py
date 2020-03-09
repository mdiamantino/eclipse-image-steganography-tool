from random import random

import cv2
import efficientnet.keras as efn
# import numpy as np
# from numba import cuda
import cupy as cp
from keras.applications.inception_v3 import preprocess_input, decode_predictions


# ============================================================

class POBA_GA:
    def __init__(self, model, image_path, T, N, alpha=2, P_c=1, P_m=0.1):
        self.model = model
        self.S = cp.array(cv2.imread(image_path))
        self.height, self.width = self.S.shape[0], self.S.shape[1]
        self.img_area = self.height * self.width
        self.maxZ_A_t0 = None
        self.T = T
        self.N = N
        self.alpha = alpha
        self.P_c, self.P_m = P_c, P_m
        self.fs = cp.zeros((N))
        self.frs = cp.zeros((N))
        self.phis = cp.zeros((N))
        self.gamma = 0.1  # TODO
        self.A, self.AS = None, None
        self.y_0, self.p_0, _, _ = self.getLables(self.S)  # True label of the adversarial image
        # Parameters to adjust the perturbation pixel mapping rule (if naked eye, pm1=10, pm2=5.8)
        self.pm1, self.pm2 = 10, 5.8
        self.mean, self.variance = 0, 0.5
        self.attacked = False
        self.A = cp.zeros((2, self.N, *self.S.shape[:2]))
        self.AS = cp.zeros((self.N, *self.S.shape[:2], 3))
        self.initialization()

    def initialization(self):
        print("\n[*] STARTING INITIALIZATION")
        random_noises_A = [self.generateRandomNoise(self.S.shape[:2]) for _ in range(self.N)]
        self.A[0] = cp.array(random_noises_A)
        self.maxZ_A_t0 = max(self.Z(i) for i in range(len(self.A[0])))
        print(self.maxZ_A_t0)
        print("[*] ENDING INITIALIZATION\n")

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
        x = cv2.resize(cp.asnumpy(np_img), dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        x = cp.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.model.predict(x)
        top2 = decode_predictions(preds, top=2)[0]
        p = [{'label': description, 'probability': probability}
             for label, description, probability in top2]
        y_1, p_1, y_2, p_2 = p[0]['label'], p[0]['probability'], p[1]['label'], p[1]['probability']
        # print(y_1, p_1)
        # if p_1 < 0.5:
        #     print(p_1)
        return y_1, p_1, y_2, p_2

    def generateRandomNoise(self, shape):
        sigma = self.variance ** 0.5
        gaussian = cp.random.normal(self.mean, sigma, shape)
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
        for t in range(self.T):
            print("Starting iteration n°", t)
            self.mergeImgPlusNoise()
            self.phis = cp.array([self.phi(i) for i in range(self.N)]) # TODO: FIX BUG
            if self.attacked:
                print("MAX---> ", cp.max(self.phis))
                # break
            self.fs = cp.divide(self.phis, cp.sum(self.phis))
            self.frs = cp.array([cp.sum(self.fs[:i + 1]) for i in range(len(self.fs))])
            # if max(self.phis) > self.gamma:
            #     optimal_t = t
            #     break
            for n in range(self.N // 2):
                i = self.selection()  # First chosen parent
                j = self.selection()  # Second chosen parent
                A_2n_1_t_plus_1, A_2n_t_plus_1 = self.crossover(self.A[0][i],
                                                                self.A[0][j])
                self.A[1, (2 * n)] = self.mutation(A_2n_1_t_plus_1)
                self.A[1, ((2 * n) + 1)] = self.mutation(A_2n_t_plus_1)
            self.A[0] = self.A[1]
            self.A[1] = cp.zeros(self.A.shape[1:])

        optimal_img_indx = cp.argmax(self.phis)
        optimal_res = self.AS[optimal_img_indx]
        print(optimal_res.tolist())
        return optimal_res

    def Z(self, i):
        """
        Calculates the perturbation size of adversarial example.
        It is a perturbation calculation indicator.
        :param A_i_t:  The tth iteration of the ith ith perturbation
        :return: Perturbation size
        """
        var = (-cp.abs(self.A[0,i]) * self.pm1 + self.pm2)
        first_exp = cp.exp(var)
        second_exp = cp.exp(self.pm2)
        perturbation_size = (1 / (1 + first_exp) - 1 / (1 + second_exp))
        return cp.sum(perturbation_size)/self.img_area

    def phi(self, i):
        """
        For a given AS_i_t, calculates its fitness function.
        :param AS_i_t: The tth iteration of the ith adversarial example
        :return:
        """
        current_adversarial_img = self.AS[i]
        y_1, p_1, y_2, p_2 = self.getLables(current_adversarial_img)
        if y_1 != self.y_0:
            print("----> Different LABELS")
            print("DIFFERENT LABELS : %s != %s" % (y_1, self.y_0))
            self.attacked = True
            res = p_1 - self.p_0 - ((self.alpha / self.maxZ_A_t0) * self.Z(i))
        else:
            res = p_2 - self.p_0
        print(res)
        return res

    def f(self, i):
        """
        For a given AS_i_t, calculates its selection probability.
        :param AS_i_t: The tth iteration of the ith adversarial example
        :return: Selection probability
        """
        return self.phis[i] / sum(self.phis[j] for j in range(self.N))

    def fr(self, i):
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
        B = cp.random.randint(2, size=(self.height, self.width))
        if random() < self.P_c:
            # print("ok")
            b_opposite = (1 - B)
            A_i_t_plus_1 = (A_i_t * B) + (A_j_t * b_opposite)
            A_j_t_plus_1 = (A_i_t * b_opposite) + (A_j_t * B)
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
        C = cp.random.randint(3, size=(self.height, self.width))
        if random() < self.P_m:
            return A_i_t_plus_1 * C
        else:
            return A_i_t_plus_1


if __name__ == "__main__":
    # self.app = ClarifaiApp(api_key='7c87e299629e4e7ea0566aca3136b214')
    # self.model = self.app.public_models.general_model
    model = efn.EfficientNetB0(weights='imagenet')  # or weights='noisy-student'
    poba_ga = POBA_GA(model, "data/test_image.jpg", 5000, 20)
    poba_ga.main()
