import cv2
from clarifai.rest import ClarifaiApp
app = ClarifaiApp(api_key='7c87e299629e4e7ea0566aca3136b214')
model = app.public_models.general_model

image = cv2.imread("data/test_image.jpg")
success, encoded_image = cv2.imencode('.png', image)
response = model.predict_by_bytes(encoded_image.tobytes())
concepts = response['outputs'][0]['data']['concepts']
y_1, p_1 = concepts[0]['name'], concepts[0]['value']
y_2, p_2 = concepts[1]['name'], concepts[1]['value']

print(y_1, p_1)
print(y_2, p_2)
