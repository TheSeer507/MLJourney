import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np


import matplotlib.pyplot as plt

tf.config.list_physical_devices('GPU')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.test.is_gpu_available(
  cuda_only=False, min_cuda_compute_capability=None
)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

