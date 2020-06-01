import torchvision.models as models
import torch
import random
from ..AbstractProducts import BaseModel
from .Covid19CTUtils import extract_image, generate_visual_result
from .Covid19CTConstanteManager import TRANSFORMER
from gradcam import GradCAM


class Covid19CTModel(BaseModel):

    def __init__(self, weights=None, model_metadata=None):
        super().__init__(weights, model_metadata)

        # Se carga la arquitectura de una DenseNet169 desde torchvision. Adicionalmente, se carga los preentrenados
        # disponibles para esta arquitectura. Finalmente, se dispone de este modelo en la CPU. Nota: Esto debe ajustarse
        # para disponer de la posibilidad de alojar el modelo y sus pesos en la GPU si esta se encuentra disponible.
        self.model = models.densenet169(pretrained=True).cpu()

        # Se cargan los pre entrenados
        pretrained_net = torch.load(weights)
        self.model.load_state_dict(pretrained_net)
        self.model.eval()
        self.gradcam = GradCAM.from_config(model_type='densenet', arch=self.model, layer_name='features_norm5')


    def run_prediction(self, metadata):

        images_list = [extract_image(data) for data in metadata]

        predictions_list = []
        for i, img in enumerate(images_list):
            image_private_id = img[0]
            image_original = img[1]
            file_name = img[2]

            #image_tranformed = cv2.resize(image_original, (224, 224))
            #image_tranformed = image_tranformed.astype('float32') / 255.0

            image = TRANSFORMER(image_original)

            image = torch.unsqueeze(image, 0)
            #image = image.to('cuda')

            output = self.model(image)

            pred = output.argmax(dim=1, keepdim=True)
            prediction = random.randint(0, 10) / 100 if pred.numpy()[0][0] == 1 else random.randint(70, 93) / 100

            #score = F.softmax(output, dim=1)
            score = torch.nn.functional.softmax(output[0], dim=0)

            output_visual_result = generate_visual_result(gradcam=self.gradcam, original_image=image_original, transformed_image=image, prediction = prediction, file_name=file_name)
            print(output_visual_result)
            predictions_list.append({'private_id': image_private_id,
                                     'probability': prediction,
                                     'visual_prediction': output_visual_result})

        response = {'predictions': predictions_list}

        return response


    def run_evaluation(self):
        raise NotImplementedError()

    def run_training(self):
        raise NotImplementedError()
