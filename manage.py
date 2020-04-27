from model import VggRes64
import pandas as pd
from keras.callbacks import TensorBoard as tb,  ModelCheckpoint

if __name__ == "__main__":

    source_dir = "./source/mtcnn_64/"

    model_name = "discriminator_2x2_dense1_label_25.h5"

    df = pd.read_csv("./source/mtcnn_64_drop_col.txt", sep=" ")
    model = VggRes64.VggRes64()
    model.build_model(64, len(df.columns)-1)

    tesnorboard = tb(log_dir='./logs/{}'.format(model_name), histogram_freq=0)
    checkpoint = ModelCheckpoint('./save/{}'.format(model_name), monitor='val_loss', verbose=1,
                                 save_best_only=True, period=10)
    model.set_tensorboard(tesnorboard)
    model.set_checkpoint(checkpoint)

    train_set = df[:-10000]
    validation_set = df[-10000:]
    generator_train = model.create_image_generator(train_set, "filename", source_dir)
    generator_validation = model.create_image_generator(validation_set, "filename", source_dir)

    model.fit_generator(generator_train, generator_validation, epochs=50)




