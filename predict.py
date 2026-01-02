import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image


# CONFIGURATION

MODEL_PATH = r"C:\Users\Sam\Desktop\Plant_Leaf_Disease_Detection\model\plant_leaf_cnn_model.h5"  # change if using .keras
IMAGE_SIZE = 128

# Class names (MUST match training order)
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two_spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___healthy'
]


# LOAD MODEL

print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")


# PREDICTION FUNCTION

def predict_image(img_path):
    if not os.path.exists(img_path):
        print("❌ Image path not found!")
        return

    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index] * 100

    print("\n🌿 Prediction Result")
    print("----------------------")
    print(f"🦠 Disease Class : {CLASS_NAMES[predicted_index]}")
    print(f"📊 Confidence    : {confidence:.2f}%")


# RUN SCRIPT

if __name__ == "__main__":
    img_path = input("\n📸 Enter image path: ").strip().strip('"').strip("'")
    predict_image(img_path)

