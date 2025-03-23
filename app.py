import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="pet_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the classes
classes = [
    'abyssinian', 'american shorthair', 'boxer', 'bulldog', 'chihuahua', 
    'corgi', 'dachshund', 'german shepherd', 'golden retriever', 'husky', 
    'labrador', 'maine coon', 'mumbai cat', 'persian cat', 'pomeranian', 
    'pug', 'ragdoll cat', 'shiba inu', 'siamese cat', 'sphynx', 'yorkshire terrier'
]

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return 'Welcome to the Pet Classifier API!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if the file is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Load and preprocess the image
            img = Image.open(file_path)
            img = img.resize((150, 150))  # Resize to match model input size
            img_array = np.array(img) / 255.0  # Rescale pixel values to [0, 1]
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Add batch dimension
            
            # Set the input tensor
            interpreter.set_tensor(input_details[0]['index'], img_array)
            
            # Run inference
            interpreter.invoke()
            
            # Get the output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class_index = np.argmax(output_data, axis=1)[0]
            predicted_class = classes[predicted_class_index]
            
            # Return the result as JSON
            return jsonify({'predicted_pet_type': predicted_class})
        
        else:
            return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render's PORT or default to 5000
    app.run(host='0.0.0.0', port=port)