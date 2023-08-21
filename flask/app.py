import os
import numpy as np
import tensorflow as tf
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, send_from_directory

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# app.py
app = Flask(__name__, template_folder='templates')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Load the trained model
model = load_model(r"D:\Brain_Tumor_Detection_From_MRI_Images\flask\uploads\brain_tumor.h5")
print("model loaded")
# Helper function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to handle the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        df= pd.read_csv(r"D:\Brain_Tumor_Detection_From_MRI_Images\patients.csv") 
        name=request.form['name']
        age=request.form["age"]
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the uploaded file to the 'uploads' folder
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image
            img = load_img(file_path, target_size=(150, 150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Use the model to make predictions
            prediction = model.predict(img_array)

            # Interpret the prediction
            if prediction >= 0.5:
                result = "Tumor detected"
                inp="tumor detected"
            else:
                result = "No tumor detected"
                inp="No tumor"
            df=df.append(pd.DataFrame({'name': [name], 'age':[age], 'status':[inp]}),ignore_index=True) 
            df.to_csv(r"D:\Brain_Tumor_Detection_From_MRI_Images\patients.csv",index = False)
            im=send_from_directory(app.config['UPLOAD_FOLDER'], filename)
            return render_template('result.html',name=name,age=age,im=im, result=result)
        else:
            return render_template('error.html', error='Invalid file format. Please upload an image (jpg, jpeg, or png).')

if __name__ == '__main__':
    app.run(debug=True)
