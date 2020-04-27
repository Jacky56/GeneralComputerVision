from model import VggRes64
import pandas as pd

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

if __name__ == "__main__":
    model_name = "discriminator_2x2_tiny_label_25.h5"

    source_dir = "./source/mtcnn_64/"
    test_dir = "./source/test_data/"
    df = pd.read_csv("./source/mtcnn_64_drop_col.txt", sep=" ")
    model = VggRes64.VggRes64()
    # model.set_model(load_model("./save/discriminator4.h5", custom_objects={"custom_loss": model.custom_loss}))
    model.set_model(load_model("./save/{}".format(model_name)))
    #
    # validation_set = df[0:100]
    # print(model.evaluate(validation_set,"filename", source_dir))
    # print(model.model.metrics_names)
    # test_set = df[0:1]
    # output = model.predict(test_set, "filename", source_dir)
    # print(np.round(output))

    ins = img_to_array(load_img(test_dir + '/ey.png'))
    p = model.model.predict(np.array([ins/255.]))

    print(p.shape)

