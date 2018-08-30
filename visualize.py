from keras.models import load_model
from keras.callbacks import TensorBoard

# Generate tensorboard logs from an existing model.
model = load_model('model.h5')
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
        write_graph=True, write_images=True)
tensorboard.set_model(model)
