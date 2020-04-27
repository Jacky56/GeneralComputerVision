from keras.layers import *
import keras.backend as K
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard

class BaseModel():

    model = None
    checkpoint = None
    tensorboard = None

    def __init__(self, model=None, checkpoint=None, tensorboard=None):

        self.model = self.set_model(model)
        self.checkpoint = self.set_checkpoint(checkpoint)
        self.tensorboard = self.set_tensorboard(tensorboard)

    def set_model(self, model):
        if model.isinstance(Model):
            self.model = model
        else:
            raise Exception("this is not Keras Model Type, type given: {}".format(type(model)))

    def set_checkpoint(self, checkpoint):
        if checkpoint.isinstance(ModelCheckpoint):
            self.checkpoint = checkpoint
        else:
            raise Exception("this is not Keras ModelCheckpoint Type, type given: {}".format(type(checkpoint)))

    def set_tensorboard(self, tensorboard):
        if tensorboard.isinstance(TensorBoard):
            self.tensorboard = tensorboard
        else:
            raise Exception("this is not Keras TensorBoard Type, type given: {}".format(type(tensorboard)))

    def draw_model(self, target):
        if self.model:
            plot_model(self.model, target)
        else:
            raise Exception("model needs to be set, use BaseModel.set_model(<model>)")

    def predict(self, data):
        if self.model:
            return self.model.predict(data)
        else:
            raise Exception("model needs to be set, use BaseModel.set_model(<model>)")

    def fit_generator(self, generator_train, generator_validation, steps_train=10, steps_validation=10, epochs=3):
        if self.model and self.checkpoint and self.tensorboard:
            callbacks = [self.tensorboard, self.checkpoint]

            return self.model.fit_generator(
                generator=generator_train,
                validation_data=generator_validation,
                epochs=epochs,
                steps_per_epoch=steps_train,
                validation_steps=steps_validation,
                shuffle=True,
                callbacks=callbacks
            )

        else:
            raise Exception(
                """
                model needs to be set, use BaseModel.set_model(<model>), model: {}
                checkpoint needs to be set, use BaseModel.set_callbacks(<ModelCheckpoint>), checkpoint: {}
                tensorboard needs to be set, use BaseModel.set_callbacks(<TensorBoard>), tensorboard: {}
                """.format(self.model, self.checkpoint, self.tensorboard)
            )

    def create_image_generator(self, df, x_col, base_directory, batch_size=32, target_size=(64,64), rotation_range=15,
                               width_shift_range=0.05,height_shift_range=0.05, shear_range=0.05, zoom_range=0.1,
                               horizontal_flip=True):

        image_generator_settings = ImageDataGenerator(
            rescale=1. / 255.,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode='nearest')

        y_col = list(df.columns).remove(x_col)
        image_generator = image_generator_settings.flow_from_dataframe(
            dataframe=df,
            directory=base_directory,
            x_col=x_col,
            y_col=y_col,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='raw'
        )

        return image_generator


if __name__ == "__main__":
    a = BaseModel()
    a.fit_generator("asd",23)

