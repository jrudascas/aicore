from ..AbstractProducts import BaseModel
import torch
from torchvision import transforms
import pandas as pd
from .ChestXNetConstanteManager import LABEL_BASELINE_PROBS, MEAN, STD
from PIL import Image


class ChestXNetModel(BaseModel):

    def __init__(self, weights=None):
        super().__init__(weights)
        checkpoint = torch.load(self.weights, map_location=lambda storage, loc: storage)
        self.model = checkpoint['model']
        self.data_transform = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN,STD)
        ])
        del checkpoint

    def run_prediction(self, data_path):
        image = Image.open(data_path)
        image = image.convert('RGB')
        image = self.data_transform(image).unsqueeze(0)

        pred = self.model(torch.autograd.Variable(image.cpu())).data.numpy()[0]
        predx = ['%.3f' % elem for elem in list(pred)]

        preds_concat = pd.concat([pd.Series(list(LABEL_BASELINE_PROBS.keys())), pd.Series(predx)], axis=1)
        preds = pd.DataFrame(data=preds_concat)
        preds.columns = ["Finding", "Predicted Probability"]
        preds.sort_values(by='Predicted Probability', inplace=True, ascending=False)
        return preds

    def run_evaluation(self):
        pass

    def run_training(self):
        pass


class OtherModel(BaseModel):
    def __init__(self, weights):
        super().__init__(weights)

    def run_prediction(self):
        return self.__class__.__name__ + " run_prediction"

    def run_evaluation(self):
        pass

    def run_training(self):
        pass