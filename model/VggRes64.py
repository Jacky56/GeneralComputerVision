from keras.layers import *
import BaseModel
from keras.models import Model
import tensorflow as tf
from keras import optimizers
class VggRes64(BaseModel.BaseModel):



    def __init__(self, model=None, checkpoint=None, tensorboard=None):
        super().__init__(model, checkpoint, tensorboard)

    def block_vgg_res(self, iamge, filters):
        conv2d1_2 = Conv2D(filters=filters,
                           kernel_size=(2,2),
                           activation='relu',
                           padding='same')(iamge)
        conv2d2_2 = Conv2D(filters=filters,
                           kernel_size=(2,2),
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
        input = Input([image_size, image_size, 3], name='image')

        block1_32 = self.block_vgg_res(input, 16)

        block2_16 = self.block_vgg_res(block1_32, 24)

        block3_8 = self.block_vgg_res(block2_16, 32)

        block4_4 = self.block_vgg_res(block3_8, 32)

        block5_2 = self.block_vgg_res(block4_4, 48)

        conv2d6_1 = Conv2D(filters=64,
                           kernel_size=(2, 2),
                           activation='relu',
                           padding='valid')(block5_2)

        flatten8_1 = Flatten()(conv2d6_1)

        dense9_64 = Dense(units=64,
                          activation='relu')(flatten8_1)

        output = Dense(units=labels,
                       activation='sigmoid')(dense9_64)

        model = Model(inputs=[input], outputs=[output])

        optimizer = optimizers.Adam()
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy', 'mse'])

        self.set_model(model)




if __name__ == "__main__":
    a = VggRes64()
    a.build_model(64, 32)
    a.draw_model("./discriminator4.png")
    a.model.summary()
    a.save_model("./discriminator4.h5")

