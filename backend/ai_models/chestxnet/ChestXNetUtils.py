import cv2
import torch
import numpy as np
from .ChestXNetConstanteManager import PATH_SAVE_VISUAL_RESPONSE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
matplotlib.use('Agg')

plt.style.use('dark_background')
plt.rc_context({'ytick.color': 'red'})


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
    probabilities = [float(item) for item in list(predictions.values())]
    sum_predictions = np.sum(probabilities)
    # instantiate cam model and get output
    label_index = np.argmax(probabilities)

    model_cam = densenet_last_layer(model)
    image_tranformed = torch.autograd.Variable(image_tranformed)
    y = model_cam(image_tranformed)
    y = y.cpu().data.numpy().squeeze()

    # pull weights corresponding to the 1024 layers from model
    weights = model.state_dict()['classifier.0.weight']
    weights = weights.cpu().numpy()

    cam = np.zeros((7, 7))

    for i, w in enumerate(weights[label_index]):
        cam += w * y[i, :, :]

    cam /= np.max(cam)

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

    img_output = np.array(image_original) if sum_predictions < 0.3 else heatmap * 0.4 + original + original_copy * 0.5

    output_path = PATH_SAVE_VISUAL_RESPONSE + file_name.split('.')[0] + '_ai_diagnosis' + '.' + file_name.split('.')[1]

    mark_paths = generate_mark(predictions)

    merge_image_mark(img_output, mark_paths, output_path)

    print(output_path)
    return output_path


def generate_mark(predictions):
    output_path_normality = PATH_SAVE_VISUAL_RESPONSE + "/chestxnet_normality.png"
    output_path_pathology = PATH_SAVE_VISUAL_RESPONSE + "/chestxnet_pathology.png"

    sum_predictions = np.sum([float(item) for item in list(predictions.values())])

    sum_predictions = 0.98 if sum_predictions >= 1 else sum_predictions

    f, ax = plt.subplots(figsize=(24, 0.6))
    plt.yticks(fontsize=30)

    df = pd.DataFrame.from_dict(
        {'Condition': ['Normality'], 'Probability': [1 - sum_predictions], 'Max': [1]}).sort_values(
        "Probability", ascending=False)
    sns.set_color_codes("pastel")
    sns.barplot(x="Max", y="Condition", data=df, color="r")
    sns.set_color_codes("dark")
    g = sns.barplot(x="Probability", y="Condition", data=df, color="r")

    for index, iter in enumerate(df.iterrows()):
        g.text(iter[1].Probability + 0.03, index + 0.20, str(round(iter[1].Probability * 100, 1)) + '%', color='red',
               ha="center", fontsize=22)

    ax.set(xlim=(0, 1))
    ax.set_xticks([])
    ax.xaxis.set_label_text("")
    ax.yaxis.set_label_text("")

    sns.despine(left=True, bottom=True)
    f.savefig(output_path_normality, dpi=400)
    plt.close(f)

    f2, ax2 = plt.subplots(figsize=(24, 2))
    plt.yticks(fontsize=30)
    df = pd.DataFrame.from_dict({'Pathology': list(predictions.keys()), 'Probability': [float(item) for item in list(predictions.values())],
                                 'Max': [1] * len(list(predictions.values()))}).sort_values("Probability",
                                                                                            ascending=False)

    df = df[:3]

    sns.set_color_codes("pastel")
    sns.barplot(x="Max", y="Pathology", data=df, color="r")
    sns.set_color_codes("dark")
    g = sns.barplot(x="Probability", y="Pathology", data=df, color="r")

    for index, iter in enumerate(df.iterrows()):
        g.text(iter[1].Probability + 0.03, index + 0.19, str(round(iter[1].Probability * 100, 1)) + '%',
               color='red',
               ha="center", fontsize=22)

    ax2.set(xlim=(0, 1))
    ax2.set_xticks([])
    ax2.xaxis.set_label_text("")
    ax2.yaxis.set_label_text("")
    sns.despine(left=True, bottom=True)
    f2.savefig(output_path_pathology, dpi=400)
    plt.close(f2)

    return output_path_normality, output_path_pathology


def merge_image_mark(image, mark_path, output_path):
    oH, oW = image.shape[:2]
    ovr = np.zeros((oH, oW, 3), dtype="uint8")
    image = np.dstack([image, np.ones((oH, oW), dtype="uint8") * 255])

    lgo_img = cv2.imread(mark_path[0], cv2.IMREAD_UNCHANGED)

    scl = math.floor((oW/lgo_img.shape[1])*100)
    w = int(lgo_img.shape[1] * scl / 100)
    h = int(lgo_img.shape[0] * scl / 100)
    dim = (w, h)

    lgo = cv2.resize(lgo_img, dim)
    lH, lW = lgo.shape[:2]

    lgo_img2 = cv2.imread(mark_path[1], cv2.IMREAD_UNCHANGED)

    w2 = int(lgo_img2.shape[1] * scl / 100)
    h2 = int(lgo_img2.shape[0] * scl / 100)
    dim2 = (w2, h2)

    lgo2 = cv2.resize(lgo_img2, dim2)
    lH2, lW2 = lgo2.shape[:2]

    ovr[oH - lH - 110:oH - 110, 15:lW + 15] = lgo[:, :, :3]
    ovr[oH - lH2 - 30:oH - 30, 15:lW2 + 15] = lgo2[:, :, :3]

    original_color_mean = np.mean(ovr, axis=2)

    final = image.copy()[:, :, :3]
    final[np.where(original_color_mean > 0)] = 0
    final = final + ovr

    cv2.putText(final, text='Stella AI Report', org=(round(lW / 2.7), final.shape[0] - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1 * (final.shape[0] / 1024), color=(0, 0, 255), thickness=2)

    cv2.imwrite(output_path, final)

    cv2.destroyAllWindows()

    return output_path
