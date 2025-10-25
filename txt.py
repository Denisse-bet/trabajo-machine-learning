import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# 1. CARGAR DATOS DE DÍGITOS
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. PREPROCESAMIENTO
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 3. MODELO CNN
model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. ENTRENAR
history = model.fit(X_train, y_train, 
                    epochs=5, 
                    validation_split=0.2,
                    batch_size=128)

# 5. EVALUAR
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Precisión en test: {test_acc:.4f}")

# 6. PREDICCIONES
y_pred = np.argmax(model.predict(X_test), axis=1)

# 7. MÉTRICAS
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 8. VISUALIZAR RESULTADOS
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.legend()

plt.subplot(1,2,2)
# Mostrar 6 predicciones de ejemplo
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(X_test[i].reshape(28,28), cmap='gray')
    plt.title(f'Real:{y_test[i]} Pred:{y_pred[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# 9. GUARDAR MODELO
model.save('modelo_digitos_tecnoforms.h5')
print("Modelo guardado para implementación en TecnoForms")
