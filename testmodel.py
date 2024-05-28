import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('/content/drive/MyDrive/my_model.h5') 
val_dir = '/content/drive/MyDrive/val'

# Sınıf isimleri
classes = ['KAPADOKYA', 'NURLU', 'SIRA']

dogru = 0
yanlis = 0
inp = input("Kaç tahmin yapılsın?")
for i in range(0,int(inp)):
	selected_class = random.choice(classes)
	class_path = os.path.join(val_dir, selected_class)
	image_name = random.choice(os.listdir(class_path))
	image_path = os.path.join(class_path, image_name)
	image_size = (224, 224)
	
	# Görüntüyü yükleme ve yeniden boyutlandırma
	img = image.load_img(image_path, target_size=image_size)
	img_array = image.img_to_array(img)
	img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekleme
	img_array = img_array / 255.0  # Normalizasyon
	
	predictions = model.predict(img_array)
	predicted_class = np.argmax(predictions, axis=1)[0]
	
	# Sınıf adlarını tahmin edilen etiketle eşleştirme
	predicted_class_name = classes[predicted_class]
	
	print(f"Selected class: {selected_class} Predicted class: {predicted_class_name}")
	if (predicted_class_name == selected_class):
		dogru+=1
	else:
		yanlis+=1
strdogru = str(dogru)
stryanlis = str(yanlis)
print(f" Doğru tahmin = {strdogru} , Yanlış tahmin = {stryanlis}")