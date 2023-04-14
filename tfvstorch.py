import time
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import tensorflow_datasets as tfds

# Load MNIST dataset for PyTorch
from torchvision import datasets, transforms

# Parameters
epochs = 5
batch_size = 128
learning_rate = 0.01

# Define the neural network model for PyTorch
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Train the PyTorch model
def train_pytorch():
    # Prepare data
    train_data = datasets.MNIST(root='./DataFiles', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = PyTorchModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"PyTorch - Epoch: {epoch + 1}/{epochs} - Loss: {loss.item()}")

# Define the neural network model for TensorFlow
class TensorFlowModel(tf.keras.Model):
    def __init__(self):
        super(TensorFlowModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28, 1))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

# Train the TensorFlow model
def train_tensorflow():
    # Prepare data
    train_data = tfds.load('mnist', split='train', as_supervised=True)
    train_data = train_data.batch(batch_size).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    # Initialize model, loss, and optimizer
    model = TensorFlowModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_data):
            with tf.GradientTape() as tape:
                output = model(data, training=True)
                loss = loss_object(target, output)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"TensorFlow - Epoch: {epoch + 1}/{epochs} - Loss: {loss.numpy()}")

if __name__ == "__main__":
    print("Training PyTorch model...")
    start_time = time.time()
    train_pytorch()
    pytorch_duration = time.time() - start_time
    print(f"PyTorch training duration: {pytorch_duration:.2f} seconds")

    print("\nTraining TensorFlow model...")
    start_time = time.time()
    train_tensorflow()
    tensorflow_duration = time.time() - start_time
    print(f"TensorFlow training duration: {tensorflow_duration:.2f} seconds")

    print("\nBenchmark completed.")