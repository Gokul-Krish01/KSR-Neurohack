import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

# Example Model (Adjust with your specific model architecture)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Assuming binary classification (adjust if needed)
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Example data (replace with your actual dataset)
# Assuming x_train, y_train, x_val, y_val are defined
# x_train, y_train = training_data
# x_val, y_val = validation_data

# Train the model and store the history object
history = model.fit(
    x_train, y_train,  # Replace with your training data
    epochs=10,         # Set the number of epochs
    batch_size=32,     # Set the batch size
    validation_data=(x_val, y_val),  # Optionally, pass validation data
    verbose=1
)

# Plotting both accuracy and loss for training and validation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Extracting accuracy values from history
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Accuracy plot
ax1.plot(train_accuracy, label='Training Accuracy')
ax1.plot(val_accuracy, label='Validation Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.legend()
ax1.grid(True)

# Extracting loss values from history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Loss plot
ax2.plot(train_loss, label='Training Loss')
ax2.plot(val_loss, label='Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.set_title('Training and Validation Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
