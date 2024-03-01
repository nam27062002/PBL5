import tensorflow as tf
model = tf.keras.models.load_model('asl_vgg16_best_weights.h5')

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model_weights.h5')
