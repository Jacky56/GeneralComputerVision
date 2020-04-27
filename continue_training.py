from model import VggRes64
import pandas as pd
from keras.callbacks import TensorBoard as tb,  ModelCheckpoint
from keras.models import load_model


if __name__ == "__main__":

    model_name = "discriminator_2x2_tiny_label_25.h5"

    source_dir = "./source/mtcnn_64/"
    df = pd.read_csv("./source/mtcnn_64_drop_col.txt", sep=" ")
    model = VggRes64.VggRes64()

    model.set_model(load_model("./save/{}".format(model_name)))
    tesnorboard = tb(log_dir='./logs/{}'.format(model_name), histogram_freq=0)
    checkpoint = ModelCheckpoint('./save/{}'.format(model_name), monitor='val_loss', verbose=1,
                                 save_best_only=True, period=10)
    model.set_tensorboard(tesnorboard)
    model.set_checkpoint(checkpoint)

    train_set = df[:-10000]
    validation_set = df[-10000:]
    generator_train = model.create_image_generator(train_set, "filename", source_dir)
    generator_validation = model.create_image_generator(validation_set, "filename", source_dir)

    model.fit_generator(generator_train, generator_validation, epochs=500)

