from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

from utils_noise import perlin


def get_noise_f(size, noise_f):
    """
    Noise generating functions with sine function mapping
    Assumes original image has shape (dim, dim, 3)
    Includes bounds for Bayesian optimization
    :param size:
    :param noise_f:
    :return:
    """
    # Gabor noise - random spread
    if noise_f == 'gabor_rand':
        pass
    # Gabor noise - uniform spread
    if noise_f == 'gabor_uni':
        pass
    # Perlin noise
    if noise_f == 'perlin':
        def noise_func(params):
            freq, freq_sine, octave = params
            noise = perlin(size, 1 / freq, int(octave), freq_sine)
            return colorize(noise)

        # Parameter boundaries for Bayesian optimization
        bounds = [{"name": "freq", "type": "continuous", "domain": (1 / 160, 1 / 20),
                   "dimensionality": 1},
                  {"name": "freq_sine", "type": "continuous", "domain": (4, 32),
                   "dimensionality": 1},
                  {"name": "octave", "type": "discrete", "domain": (1, 2, 3, 4),
                   "dimensionality": 1}]
        return noise_func, bounds


def load_predict(model_name):
    """
    Load predict function of model
    :param model_name:
    :return:
    """
    # Inception v3
    if model_name == "IncV3":
        from keras.applications.inception_v3 import InceptionV3
        from keras.applications.inception_v3 import decode_predictions, preprocess_input
        model = InceptionV3(weights="imagenet")

        def predict_prob(vec):
            img = vec.reshape((1, 299, 299, 3)).astype(np.float)
            pred = model.predict(preprocess_input(img))
            return pred[0], decode_predictions(pred, top=6)[0]
    return predict_prob


def colorize(noise, color=[1, 1, 1]):
    """
    Color noise
    :param noise: Has dimension 2 or 3, pixel range [0, 255]
    :param color: is [a, b, c] where a, b, c are from {-1, 0, 1}
    :return:
    """
    if noise.ndim == 2:  # expand to include color channels
        noise = np.expand_dims(noise, 2)
    return (noise - 0.5) * color * 2  # output pixel range [-1, 1]


def perturb(img, noise, norm):
    """
    Perturb image and clip to maximum perturbation norm
    :param img: image with pixel range [0, 1]
    :param noise: noise with pixel range [-1, 1]
    :param norm: L-infinity norm constraint
    :return: Perturbed image
    """
    noise = np.sign((noise - 0.5) * 2) * norm
    noise = np.clip(noise, np.maximum(-img, -norm), np.minimum(255 - img, norm))
    return (img + noise)


# Specify image dimensions
dim = 299

## Model & Noise
# Model and noise settings
noise_f = "perlin"  # ""perlin"
model_name = "IncV3"  # "IncV3"

# Load model and noise function
predict_prob = load_predict(model_name)
noise_func, bounds = get_noise_f(dim, noise_f)

## Constraints
max_norm = 8
max_query = 10
init_query = 5

## Initialization
# Get original image and index
image_path = "data/b.png"
img = image.load_img(image_path, target_size=(dim, dim))
orig_img = image.img_to_array(img)
probs, _ = predict_prob(orig_img.reshape((dim, dim, 3)))
orig_ind = np.argmax(probs)

# Initial queries for Bayesian optimization
np.random.seed(0)
feasible_space = GPyOpt.Design_space(space=bounds)
initial_design = GPyOpt.experiment_design.initial_design("random", feasible_space,
                                                         init_query)


# Objective function
class objective_func:
    def __init__(self):
        pass

    def f(self, params):
        params = params[0]
        payload = perturb(orig_img, noise_func(params), max_norm)
        scores, decoded = predict_prob(payload)
        orig_score = scores[orig_ind]
        return orig_score - decoded[1][2]


best_f = 1
queries = 0
obj_func = objective_func()

# Gaussian process and Bayesian optimization
objective = GPyOpt.core.task.SingleObjective(obj_func.f, num_cores=1)
model = GPyOpt.models.GPModel(exact_feval=False, optimize_restarts=5, verbose=False)
aquisition_opt = GPyOpt.optimization.AcquisitionOptimizer(feasible_space)
acquisition = GPyOpt.acquisitions.AcquisitionLCB(model, feasible_space,
                                                 optimizer=aquisition_opt)
evaluator = GPyOpt.core.evaluators.Sequential(acquisition, batch_size=1)

## Attack
BOpt = GPyOpt.methods.ModularBayesianOptimization(model, feasible_space, objective,
                                                  acquisition, evaluator, initial_design)

while queries < max_query and best_f > 0:
    queries += 1
    BOpt.run_optimization(max_iter=1)
    best_f = BOpt.fx_opt
    if queries % 5 == 0: print(
        "Query %i, Objective Function %0.2f" % (queries, best_f))  # Print every 5th query

print("Attack failed.") if best_f > 0 else print("Success!")

# Evaluate best parameters
params = BOpt.x_opt
payload = perturb(orig_img, noise_func(params), max_norm)
scores, decoded = predict_prob(payload)

# Results
print("Objective function value:", best_f)
print("Parameters:", params)

## Visualize Results
scores_orig, decoded_orig = predict_prob(orig_img.reshape((dim, dim, 3)))
# Altered image
plt.imshow(payload.astype(np.uint8))
scores_altered, decoded_altered = predict_prob(payload.reshape((dim, dim, 3)))

image.save_img("cover_image.png", payload)
