#IMPORTACIÓN DE BIBLIOTECAS
import tensorflow as tf              # Framework principal para redes neuronales
import tensorflow_datasets as tfds   # Datasets preprocesados
import matplotlib.pyplot as plt      # Visualización de gráficos
import numpy as np                   # Operaciones matemáticas

#CARGA Y PREPARACIÓN DE DATOS
# Cargar dataset MNIST (dígitos escritos a mano)
(ds_train, ds_test,), ds_info = tfds.load(
    'mnist',                        # Dataset de dígitos 0-9
    split=['train[:80%]', 'train[80%:]'],  #80% entrenamiento, 20% prueba
    with_info=True,                 # Información del dataset
    as_supervised=True,             # Formato (imagen, etiqueta)
)

# PREPROCESAMIENTO DE IMÁGENES
def preprocessing(img, label):
    # Redimensionar imágenes a 28x28 píxeles
    img = tf.image.resize(img, (28, 28))
    # Normalizar valores de píxeles [0-255] → [0-1]
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# PIPELINE DE DATOS OPTIMIZADO
# Dataset de entrenamiento /Aplicar preprocesamiento  /Mezclar datos  /Lotes de 32 imágenes  /Precargar datos
train_ds = ds_train.map(preprocessing).shuffle(1000).batch(32).prefetch(1)          

#Dataset de prueba /Aplicar preprocesamiento  /Lotes de 32 imágenes  /Precargar datos
test_ds = ds_test.map(preprocessing).batch(32).prefetch(1)           

# VISUALIZACIÓN DE DATOS
# Mostrar 6 imágenes de ejemplo del dataset
for imgs, labels in train_ds.take(1):
    for i in range(6):
        plt.subplot(2, 3, i+1)           # Cuadrícula 2x3
        plt.imshow(imgs[i])              # Mostrar imagen
        plt.title(labels[i])             # Mostrar etiqueta
        plt.axis('off')                  # Ocultar ejes
    plt.show()                           # Mostrar gráfico

# ARQUITECTURA DEL MODELO CNN
model = tf.keras.models.Sequential([
    #CAPA CONVOLUCIONAL 1
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),      #Reducción dimensional
    
    # CAPA CONVOLUCIONAL 2
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),      # Reducción dimensional
    
    # CAPA CONVOLUCIONAL 3
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),      # Reducción dimensional
    
    # TRANSICIÓN A CAPAS DENSA
    tf.keras.layers.Flatten(),           # Aplanar a 1D
    
    # CAPA DENSA OCULTA
    tf.keras.layers.Dense(64, activation='relu'),
    
    # CAPA DE SALIDA
    tf.keras.layers.Dense(10, activation='softmax')  # 10 clases (dígitos 0-9)
])

# COMPILACIÓN DEL MODELO
model.compile(
    optimizer='adam',                           # Optimizador adaptativo
    loss='sparse_categorical_crossentropy',     # Función de pérdida
    metrics=['accuracy']                        # Métrica de evaluación
)

#ENTRENAMIENTO DEL MODELO
history = model.fit(
    train_ds,                                   # Datos de entrenamiento
    epochs=5,                                   # Número de épocas
    validation_data=test_ds                     # Datos de validación
)

# VISUALIZACIÓN DEL RENDIMIENTO
#Gráfico de precisión durante entrenamiento
plt.plot(history.history['accuracy'], label='Entrenamiento')      # Precisión entrenamiento
plt.plot(history.history['val_accuracy'], label='Validación')     # Precisión validación
plt.title('Precisión del Modelo')              # Título del gráfico
plt.grid()                                     # Cuadrícula
plt.legend()                                   # Leyenda
plt.show()                                     # Mostrar gráfico

# PREDICCIÓN Y EVALUACIÓN
# Probar predicción con una imagen de prueba
for img, label in test_ds.take(1):
    pred = model.predict(img)                   # Realizar predicción
    plt.imshow(img[0])                         # Mostrar primera imagen
    plt.show()                                 # Mostrar imagen
    y_pred = model.predict(np.expand_dims(img[0], 0))  # Predecir imagen específica
    print(np.argmax(y_pred, 1))                # Mostrar predicción (dígito 0-9)