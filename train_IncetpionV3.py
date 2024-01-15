import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import os
import matplotlib.pyplot as plt


################################################################################################
#
# Mariusz Wisniewski KD-712
#
# Podstawy rozpoznawania obrazów
# 4606-PS-00000CJ-C001
# Agnieszka Jastrzębska
#
# Doszkalanie modelu InceptionV3 z dodatkowymi warstwami - wersja pierwotna
#
# Doszkolony w oparciu o 3276 zdjec podzielonych na dwie kategorie
#
# 2024-01-10
#
################################################################################################



# model 1
# link to source: https://github.com/chandrikadeb7/Face-Mask-Detection

trained_model = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

for layer in trained_model.layers:
    layer.trainable = False

last_layer = trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


path = os.getcwd()
path = os.path.join(path, 'dataset')

# declare train_path
path_train = os.path.join(path, 'train')

# declare test_path
path_validate = os.path.join(path, 'validate')

# declare test_path
path_test = os.path.join(path, 'train')   # nie ma test to jest train


train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(path_train,
                                                    batch_size=64,
                                                    class_mode='binary',
                                                    target_size=(224, 224))

validate_datagen = ImageDataGenerator(rescale=1.0/255.)

validation_generator = validate_datagen.flow_from_directory(path_validate,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(224, 224))

test_datagen = ImageDataGenerator(rescale=1.0/255.)

test_generator = test_datagen.flow_from_directory(path_test,
                                                        batch_size=1,
                                                        class_mode='binary',
                                                        target_size=(224, 224))

history = model.fit(train_generator,
          validation_data=validation_generator,
          steps_per_epoch=12,      # to sprawia ze tylko 12 batchow w kazdej epoce
          epochs=5,
          validation_steps=9,      # to sprawia ze tylko 9 batchow w kazdej epoce
          verbose=1)

# dodane rysowanie historii
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('modele/incepctionV3_0b_history.png')
plt.show()


hist_save=list(zip(epochs_range, acc, val_acc, loss, val_loss))
# hist_save = [[epochs_range[i], acc[i], val_acc[i], loss[i], val_loss[i]] for i in range(len(epochs_range))]
#print("report:")
#print(hist_save)

import csv

header = ['epoch', 'acc', 'val_acc', 'loss', 'val_loss']
with open('modele/incepctionV3_0b_history.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f, delimiter =' ')

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(hist_save)


model.predict(next(train_generator)[0])

# trzeba by jeszcze zapisac historie

model.save("modele/imagenet_inceptionV3_0b.h5")


## dodane testowanie modelu
new_model = tf.keras.models.load_model('modele/imagenet_inceptionV3_0b.h5')

# standardowa ewaluacja koncowa modelu
result = new_model.evaluate(test_generator)
print(dict(zip(new_model.metrics_names, result)))

print('modele/imagenet_inceptionV3_0b_evaluate.txt')
outres_filename='modele/imagenet_inceptionV3_0b_evaluate.txt'

with open(outres_filename, 'w') as f:
    text_out="loss: " + str(result[0]) + " accuracy: " + str(result[1])
    f.write(text_out)
