import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('model.h5')
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)



