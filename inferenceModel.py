import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, img: np.ndarray):
        img = cv2.resize(img, self.input_shapes[0][1:3][::-1])
        img_pred = np.expand_dims(img, axis=0).astype(np.float32)
        preds = self.model.run(self.output_names, {self.input_names[0]: img_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("Models/configs.yaml")
    model = ImageToModel(model_path=configs.model_path, char_list=configs.vocab)
    df = pd.read_csv("Models/val.csv").values.tolist()

    accum_cer = []
    for imag_path, label in tqdm(df):
        img = cv2.imread(imag_path.replace("\\", "/"))
        prediction = model.predict(img)
        cer = get_cer(prediction, label)
        print(f"Image: {imag_path}, Label: {label}, Prediction: {prediction}, CER: {cer}")
        accum_cer.append(cer)

        # resize by 4x
        img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4))
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Average CER: {np.average(accum_cer)}")
