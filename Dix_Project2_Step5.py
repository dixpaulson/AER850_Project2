#Step 5

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
import matplotlib.pyplot as plt
import numpy as np

# Load the model
model = load_model('DixP2.h5')

test_images = {
    "test_crack": "/Data/test/crack/test_crack.jpg",
    "test_missinghead": "/Data/test/missinghead/test_missinghead.jpg",
    "test_paintoff": "/Data/test/paintoff/test_paintoff.jpg"
}

class_labels = ["Crack", "Missing Screw-Head", "Paint Off"]


# Preprocess and predict image
def predict_image(image_path):
    img = load_img(image_path, target_size=(100, 100))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0  

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence

# Plot test images with predictions

plt.figure(figsize=(16, 4))
for x, (label, image_path) in enumerate(test_images.items()):
    predicted_class, confidence = predict_image(image_path)
    
    img = load_img(image_path, target_size=(100, 100))
    
    plt.subplot(1, 3, x + 1)
    plt.imshow(img)
    plt.title(f"{predicted_class} ({confidence:.2f})")
    plt.axis('off')

plt.tight_layout()
plt.show()