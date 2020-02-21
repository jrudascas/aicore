from ..AbstractProducts import BaseModel
import torch
from torchvision import transforms
import pandas as pd
from .ChestXNetConstanteManager import LABEL_BASELINE_PROBS, MEAN, STD
from PIL import Image
from collections import OrderedDict
import numpy as np


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
        if 'url' in metadata:
            import wget
            image_url = metadata['url']['image_url']

            if isinstance(image_url, str):
                local_image_filename = wget.download(image_url)
                image = Image.open(local_image_filename)
                image = image.convert('RGB')

                images_list.append((image_url, image))
            elif isinstance(image_url, list):
                for url in image_url:
                    local_image_filename = wget.download(url)
                    image = Image.open(local_image_filename)
                    image = image.convert('RGB')

                    images_list.append((url, image))
        elif 'bytestream' in metadata:
            orderdict = OrderedDict(metadata['bytestream'])

            for key, value in orderdict.items():
                bytes_sequence = value['stream']
                ss = np.mean(np.array(bytes_sequence), axis=2)
                image = Image.fromarray(ss)
                image = image.convert('RGB')

                images_list.append((key, image))

        prediction_list = []
        for i, img in enumerate(images_list):
            image = self.data_transform(img[1]).unsqueeze(0)

            pred = self.model(torch.autograd.Variable(image.cpu())).data.numpy()[0]
            predx = ['%.2f' % elem for elem in list(pred)]
            prediction_list.append({images_list[i][0]:dict(zip(list(LABEL_BASELINE_PROBS.keys()), predx))})

        response = {'predicctions': prediction_list}

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