import efficientnet.keras as efn
import numpy as np
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image

model = efn.EfficientNetB0(weights='imagenet')  # or weights='noisy-student'


def predict(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    top2 = decode_predictions(preds, top=2)[0]
    p = [{'label': description, 'probability': probability}
         for label, description, probability in top2]
    return p[0]['label'], p[0]['probability'], p[1]['label'], p[1]['probability']

def predictFromArray(array):
    img = image.load_img(image_file, target_size=(224, 224))
    x = image.img_to_array(array)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    top2 = decode_predictions(preds, top=2)[0]
    p = [{'label': description, 'probability': probability}
         for label, description, probability in top2]
    return p[0]['label'], p[0]['probability'], p[1]['label'], p[1]['probability']


pred = predict("output.jpg")

print()
print(pred)


