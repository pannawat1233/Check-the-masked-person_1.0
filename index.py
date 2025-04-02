from PIL import Image
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ตั้งค่าพาธโฟลเดอร์
xml_folder = "/teamspace/studios/this_studio/data/annotations"
img_folder = "/teamspace/studios/this_studio/data/images"

x_data = []
y_data = []

def parse_xml(file_path):
    """Parse XML and extract objects' bounding boxes and labels."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})
    return objects

def zoom_img(img_path, bbox):
    """Crop and resize image to focus on the bounding box and convert to RGB."""
    img = Image.open(img_path).convert("RGB")  # แปลงเป็น RGB
    xmin, ymin, xmax, ymax = bbox
    cropped = img.crop((xmin, ymin, xmax, ymax))
    zoomed = cropped.resize((224, 224), Image.Resampling.LANCZOS)
    return np.array(zoomed)  # ไม่ต้องเพิ่มมิติที่สามเหมือนขาวดำ

for file in os.listdir(xml_folder):
    file_path = os.path.join(xml_folder, file)
    img_file = file.replace(".xml", ".png")
    img_path = os.path.join(img_folder, img_file)
    
    if not os.path.exists(img_path):
        print(f"Warning: Image {img_path} not found!")
        continue
    
    objects = parse_xml(file_path)
    
    for obj in objects:
        zoomed_img = zoom_img(img_path, obj["bbox"])
        x_data.append(zoomed_img)
        y_data.append(obj['label'])

x_data = np.array(x_data) / 255.0  # Normalize

label_encoder = LabelEncoder()
y_data = label_encoder.fit_transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=20,        
    width_shift_range=0.2,    
    height_shift_range=0.2,   
    shear_range=0.2,          
    zoom_range=0.2,           
    horizontal_flip=True,     
    fill_mode='nearest'       
)

datagen.fit(x_train)

# โมเดล CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),  # ใช้ภาพ RGB
    MaxPooling2D(2,2),
    BatchNormalization(),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax') 
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(datagen.flow(x_train, y_train, batch_size=16), epochs=14, validation_data=(x_test, y_test))

model.evaluate(x_test, y_test)

model.save('Full_modelRGB.h5')

# โหลดโมเดลที่บันทึกไว้
model = load_model("Full_modelRGB.h5")
