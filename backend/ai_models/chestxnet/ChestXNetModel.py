from ..AbstractProducts import BaseModel
import torch
from torchvision import transforms
from .ChestXNetConstanteManager import LABEL_BASELINE_PROBS, MEAN, STD
from PIL import Image
import numpy as np
import wget
import validators
from os import path


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

    def run_prediction(self, metadata):
        #if not check_metadata(metadata):
            #raise Exception('metadata structure for self.__class__.__name__ is wrong. Check its documentation')

        #Cambiar por un metodo fabrica

        images_list = []

        for data in metadata:
            if 'url_path' in data:
                if validators.url(data['url_path']):
                    local_image_filename = wget.download(data['url_path'])
                    image = Image.open(local_image_filename)
                elif path.exists(data['url_path']):
                    image = Image.open(data['url_path'])
                else:
                    raise Exception('url_path wrong')
            image = image.convert('RGB')

            private_id = ''
            if 'private_id' in data:
                private_id = data['private_id']

            images_list.append((private_id, image))

        predictions_list = []
        for i, img in enumerate(images_list):
            image = self.data_transform(img[1]).unsqueeze(0)

            predictions = self.model(torch.autograd.Variable(image.cpu())).data.numpy()[0]
            sum_predictions = np.sum(predictions)
            predx = ['%.2f' % elem for elem in list(predictions)]
            predictions_list.append({'private_id':img[0],'pathology_probability': 1 if sum_predictions > 1 else sum_predictions, 'prediction_details':dict(zip(list(LABEL_BASELINE_PROBS.keys()), predx))})

        response = {'predictions':predictions_list}

        return response

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