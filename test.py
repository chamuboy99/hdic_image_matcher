import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs detected: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f" → GPU {i}: {gpu}")