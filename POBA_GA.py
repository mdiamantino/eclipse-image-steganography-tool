from random import random

import cv2
import efficientnet.keras as efn
# import numpy as np
# from numba import cuda
import cupy as cp
from keras.applications.inception_v3 import preprocess_input, decode_predictions


def predictWithCLarifai(np_img):
    success, encoded_image = cv2.imencode('.png', np_img)
    if not success:
        raise TypeError
    response = model.predict_by_bytes(encoded_image.tobytes(), is_video=False)
    concepts = response['outputs'][0]['data']['concepts']
    # Label with highest confidence of the target method for AS_i_t
    # (output label)
    y_1, p_1 = concepts[0]['name'], concepts[0]['value']
    # Label with second highest confidence of the target method for AS_i_t
    y_2, p_2 = concepts[1]['name'], concepts[1]['value']
    return y_1, p_1, y_2, p_2


def predictWithInceptionV3(np_img):
    x = cv2.resize(cp.asnumpy(np_img),
                   dsize=(224, 224),
                   interpolation=cv2.INTER_CUBIC)
    x = cp.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    top2 = decode_predictions(preds, top=2)[0]
    p = [{'label': description, 'probability': probability}
         for label, description, probability in top2]
    y_1 = p[0]['label']         # First predicted label
    p_1 = p[0]['probability']   # First confidence prediction
    y_2 = p[1]['label']         # Second predicted label
    p_2 = p[1]['probability']   # Second confidence prediction
    return y_1, p_1, y_2, p_2


class POBA_GA:
    def __init__(self, predictionMethod,
                 image_path, max_it, pop_size,
                 perturbation_ratio=3,
                 p_crossover=1,
                 p_mutation=0.3,
                 pm1=10,
                 pm2=5.8,
                 mean=0,
                 variance=1,
                 gamma=-0.3):
        """
        :param predictionMethod:
        :param image_path: Path of the selected image for adversarial attack
        :param max_it: Maximum number of iterations
        :param pop_size: Population size
        :param perturbation_ratio: Proportional coefficient used to adjust
        the proportion of attack performance and perturbation size.
        :param p_crossover: Probability of crossover (the larger, the faster
        the faster the fitness function converges)
        :param p_mutation: Probability of mutation
        :param pm1: Parameter to adjust the perturbation pixel mapping rule
        :param pm2: Parameter to adjust the perturbation pixel mapping rule
        :param mean:
        :param variance:
        :param gamma: Used to stop if fitness function result is greater
        """
        self.__getLables = predictionMethod
        self.__original_img_ = cp.array(cv2.imread(image_path))
        self.__height_ = self.__original_img_.shape[0]
        self.__width_ =self.__original_img_.shape[1]
        self.__img_area_ = self.__original_img_.shape[0] * self.__original_img_.shape[1]
        self.__max_iterations_ = max_it
        self.__population_size_ = pop_size
        self.__perturbation_ratio_ = perturbation_ratio
        self.__p_crossover_, self.__p_mutation_ = p_crossover, p_mutation
        self.__max_initial_perturbation_ = None
        self.__optimal_adv_img_ = None
        self.__last_label_ = None
        self.__fitness_values_ = cp.zeros((pop_size))
        self.__selection_probabilities_ = cp.zeros((pop_size))
        self.__cumulative_probabilities_ = cp.zeros((pop_size))
        self.__termination_condition_ = gamma
        self.__original_label_, self.__original_confidence_, _, _ = self.__getLables(self.__original_img_)
        # Parameters to adjust the perturbation pixel mapping rule (if naked eye, pm1=10, pm2=5.8)
        self.__pm1_, self.__pm2_ = pm1, pm2
        self.__mean_, self.__variance_ = mean, variance
        self.isAttacked = False
        self.__adv_perturbations_ = cp.zeros((2,
                                              self.__population_size_,
                                              *self.__original_img_.shape[:2]))
        self.__adv_examples_ = cp.zeros((self.__population_size_,
                                         *self.__original_img_.shape[:2],
                                         3))

    def __initializeRandomly(self):
        random_noises_A = [self.__generateRandomNoise(self.__original_img_.shape[:2]) for _ in range(self.__population_size_)]
        self.__adv_perturbations_[0] = cp.array(random_noises_A)
        self.__max_initial_perturbation_ = max(self.__computePerturbationSizes(i) for i in range(self.__population_size_))
        print(self.__max_initial_perturbation_)
        print("[*] INITIALIZATION COMPLETED\n")

    def __generateRandomNoise(self, random_noise_shape):
        """
        Generates a random gaussian noise image.
        :param random_noise_shape: Shape of the image (1 channel only)
        :return: Noise
        """
        sigma = self.__variance_ ** 0.5
        gaussian_noise = cp.random.normal(self.__mean_,
                                          sigma,
                                          random_noise_shape)
        return gaussian_noise

    def __mergeImgAndNoise(self):
        for channel_index in range(3):
            self.__adv_examples_[:, :, :, channel_index] = \
                self.__original_img_[:, :, channel_index] + \
                self.__adv_perturbations_[0, :, :, :]

    def main(self):
        self.__initializeRandomly()
        print("[*] STARTING ATTACK")
        for t in range(self.__max_iterations_):
            print("\t[+] Iteration", t)
            self.__mergeImgAndNoise()
            self.__fitness_values_ = cp.array([self.__computeFitnessValue(i) for i in range(self.__population_size_)])
            # if self.isAttacked:
            #     print("MAX---> ", cp.max(self.__fitness_values_))
            self.__computeSelectionProbabilities()
            self.__computeCumulativeProbabilities()
            if cp.max(self.__fitness_values_) > self.__termination_condition_:
                break
            for n in range(self.__population_size_ // 2):
                i = self.__rouletteSelection()  # First chosen parent
                j = self.__rouletteSelection()  # Second chosen parent
                A_2n_1_t_plus_1, A_2n_t_plus_1 = self.__crossover(self.__adv_perturbations_[0][i],
                                                                  self.__adv_perturbations_[0][j])
                self.__adv_perturbations_[1, (2 * n)] = self.__mutation(A_2n_1_t_plus_1)
                self.__adv_perturbations_[1, ((2 * n) + 1)] = self.__mutation(A_2n_t_plus_1)
            self.__adv_perturbations_[0] = self.__adv_perturbations_[1]
            self.__adv_perturbations_[1] = cp.zeros(self.__adv_perturbations_.shape[1:])
        print("[*] ATTACK COMPLETED WITH NEW LABEL %s\n" % self.__last_label_)
        self.__saveAdversarialImg()

    def __saveAdversarialImg(self):
        optimal_adv_img_index = cp.argmax(self.__fitness_values_)
        self.__optimal_adv_img_ = self.__adv_examples_[optimal_adv_img_index]
        cv2.imwrite("output.jpg", cp.asnumpy(self.__optimal_adv_img_))
        print("[*] ADVERSARIAL IMAGE SAVED SUCCESSFULLY\n")


    def __computePerturbationSizes(self, i):
        """
        Calculates the perturbation size of adversarial example.
        It is a perturbation calculation indicator.
        """
        var = (-cp.abs(self.__adv_perturbations_[0, i]) * self.__pm1_ + self.__pm2_)
        perturbation_size = (1 / (1 + cp.exp(var)) - 1 / (1 + cp.exp(self.__pm2_)))
        return cp.float(cp.sum(perturbation_size)/self.__img_area_)

    def __computeFitnessValue(self, i):
        """
        It evaluates the quality of adversarial example.
        :return: Quality of adv. example
        """
        current_adversarial_img = self.__adv_examples_[i]
        predicted_label, main_confidence, second_predicted_label, second_confidence = self.__getLables(current_adversarial_img)
        if predicted_label != self.__original_label_:
            self.__last_label_ = predicted_label
            # print("----> Different LABELS")
            # print("DIFFERENT LABELS : %s != %s" % (predicted_label, self.__original_label_))
            self.isAttacked = True
            res = main_confidence - self.__original_confidence_ - ((self.__perturbation_ratio_ / self.__max_initial_perturbation_) * self.__computePerturbationSizes(i))
        else:
            res = second_confidence - self.__original_confidence_
        return res

    def __computeSelectionProbabilities(self):
        """
        Calculates the selection probability of all adversarial examples.
        Probabilities are stocked to the vector.
        """
        self.__selection_probabilities_ = cp.divide(self.__fitness_values_, cp.sum(self.__fitness_values_))

    def __computeCumulativeProbabilities(self):
        """
        Calculates the cumulative probability of all adversarial examples.
        It is mandatory to produce two children
        by crossover and mutation operations.
        """
        self.__cumulative_probabilities_ = cp.array([cp.sum(self.__selection_probabilities_[:i + 1]) for i in range(len(self.__selection_probabilities_))])

    def __rouletteSelection(self):
        """
        Selects the perturbations corresponding to the adversarial examples:
        Chooses two parent examples to produce
        two children by crossover and mutation operators.
        """
        i = 0
        while i < self.__population_size_ and random() < self.__cumulative_probabilities_[i]:
            i += 1
        return i

    def __crossover(self, A_i_t, A_j_t):
        """
        Generates new examples from selected examples A_i_t, A_j_t
        :param A_i_t: Parent 1 perturbation
        :param A_j_t: Parent 2 perturbation
        :return: New examples
        """
        B = cp.random.randint(2, size=(self.__height_, self.__width_))
        if random() < self.__p_crossover_:
            b_opposite = (1 - B)
            A_i_t_plus_1 = (A_i_t * B) + (A_j_t * b_opposite)
            A_j_t_plus_1 = (A_i_t * b_opposite) + (A_j_t * B)
        else:
            A_i_t_plus_1 = A_i_t
            A_j_t_plus_1 = A_j_t
        return A_i_t_plus_1, A_j_t_plus_1

    def __mutation(self, A_i_t_plus_1):
        """
        Alters some example by a certain probability during the breeding process.
        :param A_i_t_plus_1:
        :return: Muted example
        """
        # two-dimensional matrix in crossover
        C = cp.random.randint(3, size=(self.__height_, self.__width_))
        if random() < self.__p_mutation_:
            possibly_muted = A_i_t_plus_1 * C
        else:
            possibly_muted = A_i_t_plus_1
        return possibly_muted


if __name__ == "__main__":
    # app = ClarifaiApp(api_key='7c87e299629e4e7ea0566aca3136b214')
    # model = self.app.public_models.general_model
    model = efn.EfficientNetB0(weights='imagenet')  # or weights='noisy-student'
    poba_ga = POBA_GA(predictWithInceptionV3, "data/test_image.jpg", 5000, 20)
    poba_ga.main()
