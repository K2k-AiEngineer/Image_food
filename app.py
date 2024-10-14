# import numpy as np
# import cv2 as cv
# from flask import Flask, render_template, request
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.applications import ResNet50
# from keras.applications.resnet50 import preprocess_input
# from keras.preprocessing.image import img_to_array
# from io import BytesIO
# from PIL import Image

# # Initialize Flask application
# app = Flask(__name__)

# # Load the pre-trained model
# model = Sequential()
# model.add(Dense(256, input_shape=(100352,), activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(1, activation='sigmoid'))
# model.load_weights('best_model.keras')  # Ensure this file exists in your project directory

# # Load ResNet50 model for feature extraction
# feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='flatten')

# # Function to preprocess the uploaded image
# def preprocess_image(image):
#     image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV
#     image = cv.resize(image, (224, 224))
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#     image = preprocess_input(image)
#     features = feature_extractor.predict(image)
#     return features.flatten()

# # Define routes
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     prediction_label = None
#     uploaded_image = None
#     if request.method == 'POST':
#         # Handle image upload
#         file = request.files['image']
#         if file:
#             uploaded_image = Image.open(BytesIO(file.read()))  # Read image from the uploaded file
#             features = preprocess_image(uploaded_image)  # Preprocess the image
#             prediction = model.predict(features.reshape(1, -1))  # Reshape for prediction
#             prediction_label = 'Food' if prediction[0][0] > 0.5 else 'Non-Food'
#     return render_template('index.html', prediction=prediction_label, uploaded_image=uploaded_image)

# if __name__ == '__main__':
#     app.run(debug=True)










import numpy as np
import cv2 as cv
from flask import Flask, render_template, request
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
from io import BytesIO
from PIL import Image
import base64

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model
model = Sequential()
model.add(Dense(256, input_shape=(100352,), activation='relu', kernel_initializer='he_normal'))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
model.load_weights('best_model.keras')  # Ensure this file exists in your project directory

# Load ResNet50 model for feature extraction
feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='flatten')

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV
    image = cv.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = feature_extractor.predict(image)
    return features.flatten()

# Define routes
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_label = None
    uploaded_image_data = None
    if request.method == 'POST':
        # Handle image upload
        file = request.files['image']
        if file:
            uploaded_image = Image.open(BytesIO(file.read()))  # Read image from the uploaded file
            uploaded_image_data = convert_image_to_base64(uploaded_image)  # Convert to base64
            features = preprocess_image(uploaded_image)  # Preprocess the image
            prediction = model.predict(features.reshape(1, -1))  # Reshape for prediction
            prediction_label = 'Food' if prediction[0][0] > 0.5 else 'Non-Food'
    return render_template('index.html', prediction=prediction_label, uploaded_image_data=uploaded_image_data)

def convert_image_to_base64(image):
    """Convert image to base64 string for embedding in HTML."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

if __name__ == '__main__':
    app.run(debug=True)
