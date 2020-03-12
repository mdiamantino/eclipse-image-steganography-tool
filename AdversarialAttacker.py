from random import random

import cv2
import efficientnet.keras as efn
# import numpy as np
# from numba import cuda
import cupy as cp
from keras.applications.inception_v3 import preprocess_input, \
    decode_predictions


def predictWithCLarifai(np_img):
    success, encoded_image = cv2.imencode('.png', cp.asnumpy(np_img))
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
    y_1 = p[0]['label']  # First predicted label
    p_1 = p[0]['probability']  # First confidence prediction
    y_2 = p[1]['label']  # Second predicted label
    p_2 = p[1]['probability']  # Second confidence prediction
    return y_1, p_1, y_2, p_2


class AdversarialAttacker:
    def __init__(self,
                 prediction_method,
                 image_path,
                 max_it,
                 pop_size,
                 perturbation_ratio=3,
                 p_crossover=1,
                 p_mutation=0.1,
                 pm1=10,
                 pm2=5.8,
                 mean=0,
                 variance=0.5,
                 gamma=0):
        """
        :param prediction_method:
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
        self.__getLables = prediction_method
        self.__original_img_ = cp.array(cv2.imread(image_path))
        self.__height_ = self.__original_img_.shape[0]
        self.__width_ = self.__original_img_.shape[1]
        self.__img_area_ = self.__original_img_.shape[0] * \
                           self.__original_img_.shape[1]
        self.__max_iterations_ = max_it
        self.__population_size_ = pop_size
        self.__perturbation_ratio_ = perturbation_ratio
        self.__p_crossover_, self.__p_mutation_ = p_crossover, p_mutation
        self.__max_initial_perturbation_ = None
        self.__optimal_adv_img_ = None
        self.__last_label_ = None
        self.__fitness_values_ = cp.zeros(shape=pop_size)
        self.__selection_probabilities_ = cp.zeros(shape=pop_size)
        self.__cumulative_probabilities_ = cp.zeros(shape=pop_size)
        self.__termination_condition_ = gamma
        self.__original_label_, self.__original_confidence_, _, _ = self.__getLables(
            self.__original_img_)
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

    def generate_adversarial_image(self):
        """
        MAIN METHOD
        :return:
        """
        self.__initialize_randomly()
        print("[*] STARTING ATTACK")
        for t in range(self.__max_iterations_):
            print("\t[+] Iteration", t)
            self.__merge_img_wth_noise()
            self.__fitness_values_ = cp.array(
                [self.__compute_fitness_value(i) for i in
                 range(self.__population_size_)])
            # if self.isAttacked:
            #     print("MAX---> ", cp.max(self.__fitness_values_))
            self.__compute_selection_probabilities()
            self.__compute_cumulative_probabilities()
            if cp.max(self.__fitness_values_) > self.__termination_condition_:
                break
            for n in range(self.__population_size_ // 2):
                # Parents selection
                i = self.__roulette_selection()
                j = self.__roulette_selection()

                # Crossover of parents
                A_2n_1_t_plus_1, A_2n_t_plus_1 = self.__crossover(
                    self.__adv_perturbations_[0][i],
                    self.__adv_perturbations_[0][j])

                # Mutation
                self.__adv_perturbations_[1, (2 * n)] = self.__mutation(
                    A_2n_1_t_plus_1)
                self.__adv_perturbations_[1, ((2 * n) + 1)] = self.__mutation(
                    A_2n_t_plus_1)
            self.__adv_perturbations_[0] = self.__adv_perturbations_[1]
            self.__adv_perturbations_[1] = cp.zeros(
                self.__adv_perturbations_.shape[1:])
        print(
            "[*] ATTACK COMPLETED WITH NEW LABEL '%s'\n" % self.__last_label_)
        self.__save_adversarial_img()

    # INITIALIZATION ==========================================================

    def __initialize_randomly(self):
        random_noises_A = [
            self.__generate_random_noise(self.__original_img_.shape[:2])
            for _ in range(self.__population_size_)]
        self.__adv_perturbations_[0] = cp.array(random_noises_A)
        self.__max_initial_perturbation_ = max(
            self.__compute_perturbation_sizes(i) for i in
            range(self.__population_size_))
        print(self.__max_initial_perturbation_)
        print("[*] INITIALIZATION COMPLETED\n")

    def __generate_random_noise(self, random_noise_shape):
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

    # MEASURE UTILS ===========================================================

    def __compute_perturbation_sizes(self, i):
        """
        Calculates the perturbation size of adversarial example.
        It is a perturbation calculation indicator.
        """
        var = (-cp.abs(self.__adv_perturbations_[0, i]) * self.__pm1_
               + self.__pm2_)
        perturbation_size = (1 / (1 + cp.exp(var))
                             - 1 / (1 + cp.exp(self.__pm2_)))
        return cp.float(cp.sum(perturbation_size) / self.__img_area_)

    # GENETIC SELECTION UTILS =================================================

    def __compute_fitness_value(self, i):
        """
        It evaluates the quality of adversarial example.
        :return: Quality of adv. example
        """
        current_adversarial_img = self.__adv_examples_[i]
        predicted_label, main_confidence, second_label, second_confidence = \
            self.__getLables(current_adversarial_img)
        res = main_confidence - self.__original_confidence_
        if predicted_label != self.__original_label_:
            self.__last_label_ = predicted_label
            # print("----> Different LABELS")
            # print("DIFFERENT LABELS : %s != %s" % (predicted_label, self.__original_label_))
            self.isAttacked = True
            res -= ((self.__perturbation_ratio_
                     / self.__max_initial_perturbation_)
                    * self.__compute_perturbation_sizes(i))
        return res

    def __compute_selection_probabilities(self):
        """
        Calculates the selection probability of all adversarial examples.
        Probabilities are stocked to the vector.
        """
        self.__selection_probabilities_ = cp.divide(
            self.__fitness_values_, cp.sum(self.__fitness_values_))

    def __compute_cumulative_probabilities(self):
        """
        Calculates the cumulative probability of all adversarial examples.
        It is mandatory to produce two children
        by crossover and mutation operations.
        """
        self.__cumulative_probabilities_ = cp.array([
            cp.sum(self.__selection_probabilities_[:i + 1])
            for i in range(len(self.__selection_probabilities_))])

    # GENETIC ALGORITHMS ======================================================

    def __roulette_selection(self):
        """
        Selects the perturbations corresponding to the adversarial examples:
        Chooses two parent examples to produce
        two children by crossover and mutation operators.
        """
        i = 0
        while i < self.__population_size_ \
                and random() < self.__cumulative_probabilities_[i]:
            i += 1
        return i

    def __crossover(self, adv_noise_i, adv_noise_j):
        """
        Generates new examples from selected examples A_i_t, A_j_t
        :param adv_noise_i: Parent 1 perturbation
        :param adv_noise_j: Parent 2 perturbation
        :return: New examples
        """
        B = cp.random.randint(2, size=(self.__height_, self.__width_))
        if random() < self.__p_crossover_:
            b_opposite = (1 - B)
            A_i_t_plus_1 = (adv_noise_i * B) + (adv_noise_j * b_opposite)
            A_j_t_plus_1 = (adv_noise_i * b_opposite) + (adv_noise_j * B)
        else:
            A_i_t_plus_1 = adv_noise_i
            A_j_t_plus_1 = adv_noise_j
        return A_i_t_plus_1, A_j_t_plus_1

    def __mutation(self, future_adv_noise):
        """
        Alters some ex. by a certain probability during the breeding process.
        :param future_adv_noise:
        :return: Muted example
        """
        # two-dimensional matrix in crossover
        C = cp.random.randint(3, size=(self.__height_, self.__width_))
        if random() < self.__p_mutation_:
            possibly_muted = future_adv_noise * C
        else:
            possibly_muted = future_adv_noise
        return possibly_muted

    # UTILS ===================================================================

    def __merge_img_wth_noise(self):
        for channel_index in range(3):
            self.__adv_examples_[:, :, :, channel_index] = \
                self.__original_img_[:, :, channel_index] + \
                self.__adv_perturbations_[0, :, :, :]

    def __save_adversarial_img(self):
        optimal_adv_img_index = cp.argmax(self.__fitness_values_)
        self.__optimal_adv_img_ = self.__adv_examples_[optimal_adv_img_index]
        cv2.imwrite("output.jpg", cp.asnumpy(self.__optimal_adv_img_))
        print("[*] ADVERSARIAL IMAGE SAVED SUCCESSFULLY\n")


if __name__ == "__main__":
    # ======================================
    # app = ClarifaiApp(api_key='7c87e299629e4e7ea0566aca3136b214')
    # model = self.app.public_models.general_model
    # ======================================
    # InceptionV3 model with imagenet or weights='noisy-student'
    model = efn.EfficientNetB0(weights='imagenet')
    # ======================================
    poba_ga = AdversarialAttacker(prediction_method=predictWithInceptionV3,
                                  image_path="data/test_image.jpg",
                                  max_it=5000,
                                  pop_size=20)
    poba_ga.generate_adversarial_image()
