import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# โหลดโมเดล
model = load_model("final_food_detector_model.h5")

# โหลด class_labels จากการฝึกของ ImageDataGenerator
# คาดว่า train_generator หรือ validation_generator มี class_indices
class_labels = ["Burger", "Sushi", "Pizza", "Ramen", "Dessert"]  # คุณสามารถเปลี่ยนแปลงตามนี้

# ฟังก์ชันทดสอบภาพ
def predict_image(img_path):
    # โหลดภาพและแปลงขนาดให้ตรงกับโมเดล
    img = image.load_img(img_path, target_size=(224, 224))  # ปรับขนาดเป็น 224x224
    img_array = image.img_to_array(img) / 255.0  # Normalize เป็น 0-1
    img_array = np.expand_dims(img_array, axis=0)  # เพิ่มมิติให้เป็น (1, 224, 224, 3)

    # ทำนายผล
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # ดึงค่า class index ที่มีค่าสูงสุด
    confidence = np.max(predictions)  # ค่าความมั่นใจของโมเดล

    # แสดงผลลัพธ์
    print(f"Prediction: {class_labels[predicted_class]} ({confidence:.2f})")
    return class_labels[predicted_class], confidence

# ทดสอบภาพใหม่
img_path = r"D:\University Project\Vision\contest\Dataset_for_development\Questionair Images\p60_1.jpg"  # เปลี่ยนเป็น path ของภาพที่ต้องการทดสอบ
predict_image(img_path)
