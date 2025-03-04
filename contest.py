import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ตั้งค่าพารามิเตอร์
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  # ปรับจำนวนรอบให้เหมาะสม
DATASET_PATH = r"D:\University Project\Vision\contest\Dataset_for_development\Instagram Photos"

# ตรวจสอบว่ามี GPU ใช้หรือไม่
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ✅ เพิ่ม Data Augmentation เพื่อให้โมเดลเรียนรู้ได้ดียิ่งขึ้น
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,   # หมุนภาพไม่เกิน 30 องศา
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ✅ โหลดโมเดล MobileNetV2 และใช้เป็น Feature Extractor
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# ✅ ปลดล็อกบางชั้นเพื่อทำ Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:  # ล็อก 100 ชั้นแรก
    layer.trainable = False

# ✅ สร้างโมเดลใหม่
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# ✅ ใช้ Learning Rate Scheduler เพื่อลดค่าเรียนรู้เมื่อ Training ไปนานขึ้น
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# ✅ ใช้ Early Stopping และ Save Model ที่ดีที่สุด
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_food_model.h5", monitor='val_loss', save_best_only=True, verbose=1)

# ✅ คอมไพล์โมเดล
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ เทรนโมเดล
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint, lr_scheduler]
)

# ✅ บันทึกโมเดลหลังจาก Train เสร็จ
model.save("final_food_detector_model.h5")

print("✅ Training Completed and Model Saved!")


# ✅ ฟังก์ชันทดสอบโมเดลจากรูปภาพ
def predict_food(image_path):
    model = keras.models.load_model("best_food_model.h5")  # โหลดโมเดลที่ดีที่สุด
    img = image.load_img(image_path, target_size=IMG_SIZE)  # โหลดรูปภาพ
    img_array = image.img_to_array(img)  # แปลงเป็น array
    img_array = np.expand_dims(img_array, axis=0)  # เพิ่ม batch dimension
    img_array /= 255.0  # ปรับค่าพิกเซลให้อยู่ในช่วง [0,1]

    prediction = model.predict(img_array)  # ทำนาย
    predicted_class = np.argmax(prediction)  # หาหมวดหมู่ที่มีค่าความน่าจะเป็นสูงสุด
    class_labels = list(train_generator.class_indices.keys())  # ดึง label ออกมา

    return class_labels[predicted_class]  # คืนค่าชื่อ class ที่ทำนายได้


# ✅ ตัวอย่างการทำนาย
image_path = r"D:\University Project\Vision\contest\test_image.jpg"  # ใส่ path ของภาพที่ต้องการทดสอบ
result = predict_food(image_path)
print(f"🍔 Predicted Food Category: {result}")
