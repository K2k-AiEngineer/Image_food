# 🍽️ Food and Non-Food Image Detection API  ( deferent model and other accuracy)

Welcome to the Food and Non-Food Image Detection API! This project provides an API that can classify images into "food" or "non-food" categories using a pre-trained machine learning model. The main goal is to help detect food-related content in images. The repository contains the complete codebase for deploying this API using Flask and a pre-trained Keras model. 🚀


## 📁 Folder Structure

```bash
Image_food/
├── templates/
│   └── index.html              # Basic frontend for testing the API
├── .gitattributes              # Git configuration file
├── app.py                      # Main Flask application file
├── best_model.keras            # Pre-trained Keras model for classification
├── requirements.txt            # List of dependencies and libraries
├── vercel.json                 # Configuration for Vercel deployment
├── wsgi.py                     # WSGI entry point for the application
└── README.md                   # Project documentation (this file)
```

## 📜 Description of Key Files
1. app.py: The main file for the Flask web application. It loads the pre-trained Keras model, sets up API routes, and handles image classification requests.

2. best_model.keras: The pre-trained machine learning model used for image classification. The model file size is approximately 300 MB.

3. templates/index.html: A simple HTML form to upload images and test the API through a web interface.

4. requirements.txt: Lists the Python packages needed to run the project, such as Flask and TensorFlow.

5. vercel.json: A configuration file for Vercel that specifies how to build and route requests for the application. It tells Vercel to use app.py as the source for building the project.

6. wsgi.py: A WSGI (Web Server Gateway Interface) entry point for the application. It allows the Flask app to be served by WSGI servers, making it suitable for deployment.

## ✅ Testing the Live API
1. Web Interface Testing
Open the deployed link in your web browser.
Use the form on the homepage to upload images and check if they are classified as food or non-food.
2. API Endpoint Testing
The API accepts POST requests with an image file.

Example cURL command:
curl -X POST -F 'file=@/path/to/your/image.jpg' <deployed_api_url>/predict

## 🛠️ Tech Stack
Backend: Flask (Python)
Machine Learning: TensorFlow and Keras for the model
Deployment: Vercel for live hosting

## ✨ Features
Classifies images as "food" or "non-food".
Provides both a web interface for manual testing and a live API for integration.
Supports large model files (300 MB).


## 📋 Requirements
Python Packages: Listed in requirements.txt:
Flask
TensorFlow
Pillow (for image processing)

## 🔍 How to Use
1. Run Locally:
Execute app.py to start the local server.
Open http://localhost:5000 in your web browser.

2. Deploy and Use the Live API:
Send POST requests to the /predict endpoint with the image file to classify.



