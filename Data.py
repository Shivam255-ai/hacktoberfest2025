import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------
# 1. Generate Dummy Data
# ---------------------------
# Let's create a simple dataset: y = 3x + 2 + noise
np.random.seed(42)
X = np.random.rand(100, 1)  # 100 samples, 1 feature
y = 3 * X + 2 + np.random.randn(100, 1) * 0.1  # adding small noise

# ---------------------------
# 2. Build the Model
# ---------------------------
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(1,)),  # hidden layer
    layers.Dense(1)  # output layer (no activation for regression)
])

# ---------------------------
# 3. Compile the Model
# ---------------------------
model.compile(
    optimizer='adam',
    loss='mse',       # Mean Squared Error
    metrics=['mae']   # Mean Absolute Error
)

# ---------------------------
# 4. Train the Model
# ---------------------------
history = model.fit(X, y, epochs=100, verbose=0)

# ---------------------------
# 5. Evaluate the Model
# ---------------------------
loss, mae = model.evaluate(X, y, verbose=0)
print(f"Mean Absolute Error: {mae:.4f}")

# ---------------------------
# 6. Make Predictions
# ---------------------------
X_test = np.array([[0.5], [0.8], [1.2]])
y_pred = model.predict(X_test)
print("Predictions:\n", y_pred)

# ---------------------------
# 7. Plot Training Loss
# ---------------------------
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()
