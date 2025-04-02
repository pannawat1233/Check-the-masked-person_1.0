import cv2
import numpy as np
import tensorflow as tf
import os

# โหลดโมเดล
model_path = r"D:\Code\TestCode\py\Full_modelRGB.h5"
img_folder = r"D:\Code\DataSet\Check the masked person\test"
model = tf.keras.models.load_model(model_path)

# ตรวจสอบจำนวน output class ของโมเดล
print("Model output shape:", model.output_shape)

for file in os.listdir(img_folder):
           
    file_path = os.path.join(img_folder, file)
    
    # โหลดภาพในรูปแบบ RGB
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # โหลดเป็นภาพสี
    if image is None:
        print(f"Error: Cannot load image from {file_path}")
        exit()

    # แปลงจาก BGR (cv2) เป็น RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ปรับขนาดให้ตรงกับโมเดล
    image = cv2.resize(image, (224, 224))

    # ทำให้เป็น float และ normalize ค่า
    image = image / 255.0

    # เพิ่ม batch dimension (batch, height, width, channels)
    image = np.expand_dims(image, axis=0)

    # ทำนายผล
    predictions = model.predict(image)
    print("Raw predictions:", predictions)

    # หา index ของค่าที่มากที่สุด
    predicted_class = np.argmax(predictions, axis=1)
    print("Predicted class index:", predicted_class)

    # ตรวจสอบว่าคลาสที่ทำนายอยู่ในช่วงที่ถูกต้องหรือไม่
    class_labels = ["No Mask", "Mask", "No Mask"]  # ปรับตามโมเดลของคุณ
    if predicted_class[0] < len(class_labels):
        print(f"Prediction: {class_labels[predicted_class[0]]}")
        print(file)
    else:
        print(f"Error: Predicted class index {predicted_class[0]} is out of range.")
