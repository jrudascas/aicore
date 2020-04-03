import cv2
import torch
import numpy as np
from .ChestXNetConstanteManager import LABEL_BASELINE_PROBS, PATH_SAVE_VISUAL_RESPONSE


class densenet_last_layer(torch.nn.Module):
    def __init__(self, model):
        super(densenet_last_layer, self).__init__()
        self.features = torch.nn.Sequential(
            *list(model.children())[:-1]
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.relu(x, inplace=True)
        return x

def generate_visual_result(predictions, model, image_tranformed, image_original, file_name):
    sum_predictions = np.sum(predictions)
    # instantiate cam model and get output
    label_index = np.argmax(predictions)

    model_cam = densenet_last_layer(model)
    image_tranformed = torch.autograd.Variable(image_tranformed)
    y = model_cam(image_tranformed)
    y = y.cpu().data.numpy().squeeze()

    # pull weights corresponding to the 1024 layers from model
    weights = model.state_dict()['classifier.0.weight']
    weights = weights.cpu().numpy()

    bias = model.state_dict()['classifier.0.bias']
    bias = bias.cpu().numpy()

    cam = np.zeros((7, 7))


    for i, w in enumerate(weights[label_index]):
        cam += w * y[i, :, :]

    cam /= np.max(cam)

    #for i in range(0, 7):
    #    for j in range(0, 7):
    #        for k in range(0, 1024):
    #            cam[i, j] += y[k, i, j] * weights[label_index, k]
    #cam += bias[label_index]

    #cam = 1 / (1 + np.exp(-cam))

    #cam = cam / list(LABEL_BASELINE_PROBS.items())[label_index][1]

    # take log
    #cam = np.log(cam)

    original = np.array(image_original)
    cam = cv2.resize(cam, original.shape[:2])

    if cam.shape != original.shape[:2]:
        cam = np.transpose(cam)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.3)] = 0
    original_copy = original.copy()
    original_color_mean = np.mean(heatmap, axis=2)
    original[np.where(original_color_mean > 0)] = 0

    original_copy[np.where(original_color_mean == 0)] = 0

    text = 'Stella AI Report: {} -- Probability = {}%'
    if sum_predictions < 0.3:
        img_output = np.array(image_original)
        text = text.format('NORMAL', round((1 - sum_predictions) * 100, 2))
    else:
        img_output = heatmap * 0.4 + original + original_copy * 0.5
        text = text.format(list(LABEL_BASELINE_PROBS.items())[label_index][0].upper(),
                           round(predictions[label_index] * 100), 2)

    cv2.putText(img_output, text=text, org=(5, img_output.shape[0] - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1 * (img_output.shape[0] / 1024), color=(0, 0, 255), thickness=2)

    output_path = PATH_SAVE_VISUAL_RESPONSE + file_name.split('.')[0] + '_ai_diagnosis' + '.' + file_name.split('.')[1]

    cv2.imwrite(output_path, img_output)

    return output_path
