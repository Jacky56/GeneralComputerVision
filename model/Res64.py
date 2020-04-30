from keras.layers import *
import BaseModel
from keras.models import Model
import tensorflow as tf
from keras import optimizers

class Res64(BaseModel.BaseModel):
    def __init__(self, model=None, checkpoint=None, tensorboard=None):
        super().__init__(model, checkpoint, tensorboard)

    def block_res(self, iamge, filters, size=(3,3)):

        conv2d1_2 = Conv2D(filters=filters,
                           kernel_size=size,
                           activation='relu',
                           padding='same')(iamge)
        conv2d2_2 = Conv2D(filters=filters,
                           kernel_size=size,
                           activation='relu',
                           padding='same')(conv2d1_2)

        upsample1 = Conv2D(filters=filters,
                           kernel_size=(1,1),
                           activation='relu',
                           padding='same')(iamge)

        add1 = add([conv2d2_2, upsample1])

        max_pool1 = MaxPooling2D()(add1)

        norm1 = BatchNormalization()(max_pool1)

        return norm1

    def custom_loss(self, y_true, y_pred):
        return tf.keras.losses.BinaryCrossentropy(y_true, y_pred, label_smoothing=0.05)

    def build_model(self, image_size, labels):
        drop_rate = 0.33

        input = Input([image_size, image_size, 3], name='image')

        block1_32 = self.block_res(input, 16)

        block2_16 = self.block_res(block1_32, 32)

        block3_8 = self.block_res(block2_16, 48)

        block3_8 = Dropout(drop_rate)(block3_8)

        block4_4 = self.block_res(block3_8, 64)

        block4_4 = Dropout(drop_rate)(block4_4)

        block5_2 = self.block_res(block4_4, 64)

        block5_2 = Dropout(drop_rate)(block5_2)

        conv2d6_1 = Conv2D(filters=96,
                           kernel_size=(2, 2),
                           activation='relu',
                           padding='valid')(block5_2)

        flatten8_1 = Flatten()(conv2d6_1)

        flatten8_1 = Dropout(drop_rate)(flatten8_1)

        dense9 = Dense(units=128,
                          activation='relu')(flatten8_1)

        dense9 = Dropout(drop_rate)(dense9)

        dense10 = Dense(units=128,
                          activation='relu')(dense9)

        dense10 = Dropout(drop_rate)(dense10)

        output = Dense(units=labels,
                       activation='tanh')(dense10)

        model = Model(inputs=[input], outputs=[output])

        optimizer = optimizers.Adam()
        model.compile(optimizer=optimizer, loss="squared_hinge", metrics=['accuracy', 'mse'])

        self.set_model(model)




if __name__ == "__main__":
    a = Res64()
    a.build_model(64, 32)
    a.draw_model("./Res64.png")
    a.model.summary()
    a.save_model("./Res64.h5")

