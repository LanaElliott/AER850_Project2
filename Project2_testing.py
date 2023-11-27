import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Load the trained models
model1 = load_model('Model_1.h5')
model2 = load_model('Model_2.h5')

# Data generator for test images

test_datagen = ImageDataGenerator(rescale=1./255)

# Define a mapping dictionary for class indices to size labels
size_mapping = {
    0: "Large",
    1: "Medium",
    2: "None",
    3: "Small"
}

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(100, 100))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to predict the class using the trained models
def predict_class(model, image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    # Find the class with the maximum probability
    predicted_class = np.argmax(prediction)
    return predicted_class

# Test image paths (replace these with your actual test images)
test_image_paths = [
    'C:/Users/rmbar/OneDrive/Documents/GitHub/AER850_Project2/Data/Test/Medium/Crack__20180419_06_19_09,915.bmp',
    'C:/Users/rmbar/OneDrive/Documents/GitHub/AER850_Project2/Data/Test/Large/Crack__20180419_13_29_14,846.bmp'
]

# Predict and print the classes for each test image using Model 1
print("Predictions using Model 1:")
for image_path in test_image_paths:
    predicted_class = predict_class(model1, image_path)
    print(f"Image: {image_path}, Predicted Class: {predicted_class}")

# Predict and print the classes for each test image using Model 2
print("\nPredictions using Model 2:")
for image_path in test_image_paths:
    predicted_class = predict_class(model2, image_path)
    print(f"Image: {image_path}, Predicted Class: {predicted_class}")

# Plot the Results
for image_path in test_image_paths:
    # Load the image for display
    img = load_img(image_path, target_size=(100, 100))
    
    # Predict the class
    predicted_class = predict_class(model1, image_path)
    
    # Get the corresponding size label from the mapping
    size_label = size_mapping[predicted_class]
    
    # Display the image with the predicted class
    plt.figure()
    plt.imshow(img)
    plt.title(f'Predicted Class: {size_label}')
    
    # Display predicted probabilities
    pred_percs = model1.predict(preprocess_image(image_path))[0] * 100
    pred_percs = np.round(pred_percs, 2)
    pred_percs_strin = ', '.join(map(str, pred_percs))
    plt.text(7, 7, f'Predicted Probabilities (Large, Medium, None, Small):\n                                       ({pred_percs_strin})%', color='yellow', fontsize=9 )
    plt.show()