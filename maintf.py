import tensorflow as tf
import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(4, input_shape=(2,), activation='sigmoid'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x, y, epochs=10000)

predictions = model.predict(x)
print(predictions)
