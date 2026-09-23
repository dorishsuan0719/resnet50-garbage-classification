import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

#  6 類標籤
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

print("載入模型")
model = tf.keras.models.load_model('garbage_resnet50.keras')
print("模型載入完成！")

# 上傳圖片
img_path = input("請輸入要預測的圖片路徑（或拖曳圖片到此視窗）： ")

# 圖片前處理

def load_and_prep_image(path, img_size=(224, 224)):
    img = image.load_img(path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # 正規化
    return np.expand_dims(img_array, axis=0), img

img_batch, img_display = load_and_prep_image(img_path)

# 模型預測
pred = model.predict(img_batch)
pred_class = np.argmax(pred, axis=1)[0]
confidence = np.max(pred)

print("\n 預測結果：")
print(f"➡ 類別：{class_names[pred_class]}")
print(f"➡ 信心度：{confidence:.2f}")


# 顯示圖片與預測結果
plt.imshow(img_display)
plt.title(f"預測：{class_names[pred_class]}  ({confidence:.2f})")
plt.axis('off')
plt.show()
