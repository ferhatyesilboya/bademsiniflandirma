import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Verileri yükleme
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')  # Üç sınıf olduğu için 'categorical' kullanılır

validation_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')  # Üç sınıf olduğu için 'categorical' kullanılır

# Modeli oluşturma
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))  # Filtre sayısını 224'ten 64'e düşürdük
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Üç sınıf olduğu için 3 nöron ve 'softmax' aktivasyonu kullanılır

# Modeli derleme
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Modeli kaydetme
model.save('/content/drive/MyDrive/my_model.h5')
