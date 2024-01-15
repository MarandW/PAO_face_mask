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
# Doszkalanie modelu InceptionV3 z dodatkowymi warstwami - wersja poprawiona
#
# Doszkolony w oparciu o 5551 zdjec podzielonych na dwie kategorie
#
# 2024-01-10
#
################################################################################################

# 528.02   -> CUDA 12.0 i nowsze
# 384 rdzen
# NVCUDA64.dll 12.0.113 -- już jest
# TensorFlow
#

path = os.getcwd()
path = os.path.join(path, 'dataset_big')
#path = os.path.join(path, 'dataset')
## https://datasetninja.com/face-mask-detection

##
# 0 - InceptionV3 pierwotny z nowymi danymi

MODEL = 0

# InceptionV3
if MODEL == 0:
    EPOCHS = 50
    BATCH_SIZE = 64
    
    BATCH_SIZE_VALIDATION = 20
    DROPOUT = 0.2
    LEARNING_RATE = 0.0001
    NUM_CLASSES = 2
    IMG_SIZE = (224, 224)     # InceptionV3 ,MobileNetV2, ResNet50, ResNet152V2
    IMG_SHAPE = IMG_SIZE + (3,)
    num_layers_fine_tune = 0  # Ile ostatnich warstw ma byc szkolonych
    NAME = 'IncepctionV3_0_nowytest_02'
    trained_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet') # dobrze classifier_activation="softmax"

# InceptionV3 ze szkoleniem 4 ostatnich warstw
elif MODEL == 1:
    EPOCHS = 50
    BATCH_SIZE = 64
    BATCH_SIZE_VALIDATION = 20
    DROPOUT = 0.5
    LEARNING_RATE = 0.0001
    NUM_CLASSES = 2
    IMG_SIZE = (224, 224)     # InceptionV3 ,MobileNetV2, ResNet50, ResNet152V2
    IMG_SHAPE = IMG_SIZE + (3,)
    num_layers_fine_tune = 0 # Ile ostatnich warstw ma byc szkolonych
    NAME = 'IncepctionV3_nowymod1_big_02'
    trained_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet') # dobrze classifier_activation="softmax"

# InceptionV3 ze szkoleniem 3 ostatnich warstw
elif MODEL == 2:
    EPOCHS = 5
    BATCH_SIZE = 64
    BATCH_SIZE_VALIDATION = 20
    DROPOUT = 0.2
    LEARNING_RATE = 0.0001
    NUM_CLASSES = 2
    IMG_SIZE = (224, 224)     # InceptionV3 ,MobileNetV2, ResNet50, ResNet152V2
    IMG_SHAPE = IMG_SIZE + (3,)
    num_layers_fine_tune = 32  # Ile ostatnich warstw ma byc szkolonych
    NAME = 'IncepctionV3_nowymod2_big_01'
    trained_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet') # dobrze classifier_activation="softmax"

elif MODEL == 3:
    EPOCHS = 5
    BATCH_SIZE = 64
    BATCH_SIZE_VALIDATION = 20
    DROPOUT = 0.3
    LEARNING_RATE = 0.0001
    NUM_CLASSES = 2
    IMG_SIZE = (224, 224)     # InceptionV3 ,MobileNetV2, ResNet50, ResNet152V2
    IMG_SHAPE = IMG_SIZE + (3,)
    num_layers_fine_tune = 0  # Ile ostatnich warstw ma byc szkolonych
    NAME = 'MobileNetV2_nowymod1_02'
    trained_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet') # dobrze classifier_activation="softmax"

elif MODEL == 4:
    EPOCHS = 5
    BATCH_SIZE = 64
    BATCH_SIZE_VALIDATION = 20
    DROPOUT = 0.3
    LEARNING_RATE = 0.0001
    NUM_CLASSES = 2
    IMG_SIZE = (224, 224)     # InceptionV3 ,MobileNetV2, ResNet50, ResNet152V2
    IMG_SHAPE = IMG_SIZE + (3,)
    num_layers_fine_tune = 0  # Ile ostatnich warstw ma byc szkolonych
    NAME = 'MobileNetV2_nowymod2_01'
    trained_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet') # dobrze classifier_activation="softmax"

else:
    EPOCHS = 3
    BATCH_SIZE = 64
    BATCH_SIZE_VALIDATION = 20
    DROPOUT = 0.2
    LEARNING_RATE = 0.0001
    NUM_CLASSES = 2
    #IMG_SIZE = (160, 160)
    #IMG_SIZE = (299, 299)    # InceptionResNetV2
    IMG_SIZE = (224, 224)     # InceptionV3 ,MobileNetV2, ResNet50, ResNet152V2
    IMG_SHAPE = IMG_SIZE + (3,)
    num_layers_fine_tune = 4  # Ile ostatnich warstw ma byc szkolonych
    NAME = 'big'
    trained_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet') # dobrze classifier_activation="softmax"

print("MODEL: ", MODEL)
print("NAME  :", NAME)

# modele
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md


#trained_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet') # dobrze classifier_activation="softmax"

# te modele z jakiegos powodu nie chca sie szkolic i jest 50%
#trained_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,include_top=False, weights='imagenet')  # fatalnie classifier_activation="softmax"
#trained_model = tf.keras.applications.ResNet152V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')  # zatkało komputer classifier_activation="softmax"
#trained_model = tf.keras.applications.EfficientNetB0(input_shape=None, weights="imagenet") # fatalnie classifier_activation="softmax"
#trained_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE, weights="imagenet") # słabo i wolno classifier_activation="softmax"
#trained_model = tf.keras.applications.EfficientNetV2S(input_shape=IMG_SHAPE, weights="imagenet", include_preprocessing=True) # słabo i wolno classifier_activation="softmax"

## https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

# ustawianie warstw nie do szkolenia

num_layers = len(trained_model.layers)
for layer in trained_model.layers[:num_layers - num_layers_fine_tune]:
#    print(f"FREEZING LAYER: {layer}")
#    layer.trainable = False
#for layer in trained_model.layers:
    layer.trainable = False

#trained_model.trainable = False

# podsumowanie modelu
trained_model.summary()

print("\n")
print(f"Warstwy modelu do doszkolenia:", num_layers_fine_tune)
print("\n")

# Dodajemy warstwy do nauczenia

if MODEL == 0:
    last_layer = trained_model.get_layer('mixed7')
#    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    x = layers.Flatten()(last_output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(NUM_CLASSES-1, activation='sigmoid')(x)

elif MODEL == 1:
    # ostatnia warstwa istniejącego modelu
    last_layer = trained_model.layers[num_layers-1]

    # punkt doklejania nowych warstw
    x = last_layer.output

    x = layers.Flatten()(x)
#    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(DROPOUT)(x)

    # The final `Dense` layer with the number of classes. - nie dziala jak sie poda 2
    x = layers.Dense(NUM_CLASSES-1, activation='sigmoid')(x)

elif MODEL == 2:
    # ostatnia warstwa istniejącego modelu
    last_layer = trained_model.layers[num_layers - 1]

    # punkt doklejania nowych warstw
    x = last_layer.output

    x = layers.Flatten()(x)
    #    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(DROPOUT)(x)

    # The final `Dense` layer with the number of classes. - nie dziala jak sie poda 2
    x = layers.Dense(NUM_CLASSES - 1, activation='sigmoid')(x)

elif MODEL == 3:
    # ostatnia warstwa istniejącego modelu
    last_layer = trained_model.layers[num_layers - 1]

    # punkt doklejania nowych warstw
    x = last_layer.output

    x = layers.Flatten()(x)
    #    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(DROPOUT)(x)

    # The final `Dense` layer with the number of classes. - nie dziala jak sie poda 2
    x = layers.Dense(NUM_CLASSES - 1, activation='sigmoid')(x)

elif MODEL == 4:
    last_layer = trained_model.get_layer('block_14_expand_relu')
    #    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    # punkt doklejania nowych warstw
    x = last_layer.output

 #   x = layers.AveragePooling2D(pool_size=(5, 5))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(DROPOUT)(x)

    # The final `Dense` layer with the number of classes.
    x = layers.Dense(1, activation='sigmoid')(x)

elif MODEL == 5:
    x = trained_model(trained_model.input, training=False)
#    x = layers.AveragePooling2D()(x)
    x = layers.AveragePooling2D(pool_size=(5, 5))(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

else:
    #inputs = tf.keras.Input(shape=IMG_SHAPE)
    #x = tf.keras.applications.inception_v3.preprocess_input(inputs)
    #x = trained_model(x)

    # ostatnia warstwa istniejącego modelu
    last_layer = trained_model.layers[num_layers-1]

    # wskazanie konkretnej nazwy gdzie model zostanie odcięty.
    #last_layer = trained_model.get_layer('mixed10')
    #last_layer = trained_model.get_layer('mixed7')

    # inny sposob na ostatnia warstwe ale czasem cos nie dziala
    #x = trained_model.output

    # punkt doklejania nowych warstw
    x = last_layer.output
    #print('x layer output shape: ', x.output_shape)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(DROPOUT)(x)

    #x = layers.GlobalAveragePooling2D()(x)

    # The final `Dense` layer with the number of classes. - nie dziala jak sie poda 2
    #x = layers.Dense(NUM_CLASSES-1, activation='softmax')(x)
    x = layers.Dense(NUM_CLASSES-1, activation='sigmoid')(x)

# The final model (wsad i koniec).
model = Model(trained_model.input, x)



# https://www.kaggle.com/code/viratkothari/face-mask-detection-on-images
#headModel = baseModel.output
#headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
#headModel = Flatten(name="flatten")(headModel)
#headModel = Dense(128, activation="relu")(headModel)
#headModel = Dropout(0.5)(headModel)
#headModel = Dense(2, activation="softmax")(headModel)

model.compile(optimizer = RMSprop(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])
#model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#              metrics=[tf.keras.metrics.BinaryAccuracy()])


model.summary()


# declare train_path
path_train = os.path.join(path, 'train')

# declare validate_path
path_validate = os.path.join(path, 'validate')

# declare test_path
path_test = os.path.join(path, 'test')


train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
#                                   brightness_range=(-10,10),
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(path_train,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary',
                                                    target_size=IMG_SIZE,
#                                                    color_mode='rgb',
#                                                    seed=1
                                                    )

validate_datagen = ImageDataGenerator(rescale=1.0/255.)

validation_generator = validate_datagen.flow_from_directory(path_validate,
                                                        batch_size=BATCH_SIZE_VALIDATION,
                                                        class_mode='binary',
                                                        target_size=IMG_SIZE,
#                                                        color_mode='rgb',
#                                                        seed=1
                                                        )

test_datagen = ImageDataGenerator(rescale=1.0/255.)

test_generator = test_datagen.flow_from_directory(path_test,
                                                        batch_size=1,
                                                        class_mode='binary',
                                                        target_size=IMG_SIZE,
#                                                        color_mode='rgb',
#                                                        seed=1
                                                  )

history = model.fit(train_generator,
          validation_data=validation_generator,
#          steps_per_epoch=12,   # steps_per_epoch = TotalTrainingSamples / TrainingBatchSize  ale te 12 przyspiesza badanie kodu
          epochs=EPOCHS,
#          validation_steps=9,   # validation_steps = TotalvalidationSamples / ValidationBatchSize
          verbose=1)


print(NAME + '_imagenet.h5')
model_filename = NAME + '_imagenet.h5'

model.save('modele/' + model_filename)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

figname = "modele/" + NAME + '_history.png'

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
plt.savefig(figname)
plt.show()

hist_save=list(zip(epochs_range, acc, val_acc, loss, val_loss))
# hist_save = [[epochs_range[i], acc[i], val_acc[i], loss[i], val_loss[i]] for i in range(len(epochs_range))]
#print("report:")
#print(hist_save)


import csv

print(NAME + '_history.csv')
history_filename = "modele/" + NAME + '_history.csv'

header = ['epoch', 'acc', 'val_acc', 'loss', 'val_loss']
with open(history_filename, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f, delimiter ='\t')

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(hist_save)

#model.predict(next(train_generator)[0])


new_model = tf.keras.models.load_model(model_filename)

# standardowa ewaluacja koncowa modelu
result = new_model.evaluate(test_generator)
outres= dict(zip(new_model.metrics_names, result))
print(outres)

print(NAME + '_evaluate.txt')
outres_filename="modele/" + NAME + '_evaluate.txt'

with open(outres_filename, 'w') as f:
    text_out="loss: " + str(result[0]) + " accuracy: " + str(result[1])
    f.write(text_out)

# testowanie modelu
#predictions = new_model.predict(test_generator)
#print(predictions)





#rerect_sized = cv2.resize(face_img, (224, 224))
#normalized = rerect_sized / 255.0
#reshaped = np.reshape(normalized, (1, 224, 224, 3))
#reshaped = np.vstack([reshaped])
#result = imagenet.predict(reshaped)
#print(result)



## https://learnopencv.com/fine-tuning-pre-trained-models-tensorflow-keras/
## https://www.tensorflow.org/tutorials/images/classification?hl=pl
## https://www.tensorflow.org/tutorials/images/transfer_learning?hl=pl
## https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub?hl=pl
## https://www.tensorflow.org/guide/keras/training_with_built_in_methods
## https://www.geeksforgeeks.org/facemask-detection-using-tensorflow-in-python/
## https://towardsdatascience.com/what-is-the-best-input-pipeline-to-train-image-classification-models-with-tf-keras-eb3fe26d3cc5
## https://github.com/tensorflow/models/tree/master/research/object_detection
## https://www.ejtech.io/learn/tflite-object-detection-model-comparison
