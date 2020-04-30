from model import VggRes64
import pandas as pd

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

if __name__ == "__main__":
    model_name = "Res_267k_64_label_25.h5"

    source_dir = "E:/Data/mtcnn_64/"
    test_dir = "./source/test_data/"
    df = pd.read_csv("./source/mtcnn_drop_col_64.txt", sep=" ")
    model = VggRes64.VggRes64()
    # model.set_model(load_model("./save/discriminator4.h5", custom_objects={"custom_loss": model.custom_loss}))
    model.set_model(load_model("./save/{}".format(model_name)))

    # validation_set = df[0:100]
    # print(model.evaluate(validation_set,"filename", source_dir))
    # print(model.model.metrics_names)
    # test_set = df[0:3]
    # print(test_set.columns)
    # print(test_set["filename"])
    # output = model.predict(test_set, "filename", source_dir)
    # for out in output:
    #     print([1 if e > 0 else -1 for e in out])
    #
    ins = img_to_array(load_img(test_dir + '/ye.png'))
    p = model.model.predict(np.array([ins/255.]))

    print(p)
    print([1 if e > 0 else -1 for e in p[0]])

