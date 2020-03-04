
import keras
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import efficientnet.keras as efn
import numpy as np
import tensorflow as tf

model = efn.EfficientNetB0(weights='imagenet')  # or weights='noisy-student'
graph = tf.get_default_graph()


def predict(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)

    global graph
    with graph.as_default():
        preds = model.predict(x)
    top2 = decode_predictions(preds,top=2)[0]
    predictions = [{'label': description, 'probability': probability}
                    for label,description, probability in top2]
    return predictions


pred = predict("/home/mdc/PycharmProjects/eclipse/data/test_image.jpg")

print()
print(pred)


