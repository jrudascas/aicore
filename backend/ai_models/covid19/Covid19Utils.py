import cv2
from .Covid19ConstanteManager import COVID19_PATH_SAVE_VISUAL_RESPONSE


def generate_visual_result(prediction, original_image, file_name):

    text = 'Stella AI Report: {} -- Probability = {}%'.format(prediction[0], round(prediction[1]*100, 2))
    cv2.putText(original_image, text=text, org=(5, original_image.shape[0] - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1 * (original_image.shape[0] / 1024), color=(0, 0, 255), thickness=2)

    output_path = COVID19_PATH_SAVE_VISUAL_RESPONSE + file_name.split('.')[0] + '_ai_diagnosis' + '.' + file_name.split('.')[1]

    cv2.imwrite(output_path, original_image)
    print(output_path)
    return output_path
