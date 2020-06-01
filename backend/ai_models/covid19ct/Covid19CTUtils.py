import wget
import validators
import cv2
import numpy as np
from .Covid19CTConstanteManager import COVID19_CT_PATH_SAVE_VISUAL_RESPONSE
from PIL import Image
from gradcam.utils import visualize_cam
from os import path


def extract_image(data):
    if 'url_path' in data:
        if validators.url(data['url_path']):
            local_image_filename = wget.download(data['url_path'])
            image = Image.open(local_image_filename).convert('RGB')
        elif path.exists(data['url_path']):
            image = Image.open(data['url_path']).convert('RGB')
        else:
            raise Exception('url_path wrong')

    private_id = ''
    if 'private_id' in data:
        private_id = data['private_id']
    image_name = data['url_path'].split('/')[-1]

    return (private_id, image, image_name)


def generate_visual_result(gradcam, original_image, transformed_image, prediction, file_name):
    output_filename_visual_response = COVID19_CT_PATH_SAVE_VISUAL_RESPONSE + file_name
    mask, _ = gradcam(
        transformed_image)  # Create a GradCAM(Gradient-weighted Class Activation Mapping) based on http://gradcam.cloudcv.org/
    heatmap, result = visualize_cam(mask, transformed_image)  # Based on the mask is created a heatmap visual response

    mask2 = np.zeros((224, 224))
    mask2[...] = mask[0, 0, :, :]

    # Changing the tensor shape to a numpy array in a standar image shape
    heatmap = heatmap.numpy()
    np_heatmap = np.zeros((heatmap.shape[1], heatmap.shape[2], heatmap.shape[0]))

    np_heatmap[..., 0] = heatmap[0, ...]
    np_heatmap[..., 1] = heatmap[1, ...]
    np_heatmap[..., 2] = heatmap[2, ...]

    #Image.fromarray(np.uint8(np_heatmap*255)).convert("RGBA").show()
    #Image.fromarray(np.uint8(original_image)).convert("RGBA").show()

    np_original_image = np.array(original_image)

    np_mask_resized = cv2.resize(mask2, np_original_image.shape[:2])

    np_heatmap_resized = cv2.resize(np_heatmap, np_original_image.shape[:2])
    np_heatmap_resized = np.uint8(255 * np_heatmap_resized)



    original_copy = np_original_image.copy()

    np_mask_resized[np.where(np_mask_resized < 0.3)] = 0

    np_heatmap_resized[np.where(np_mask_resized == 0)] = 0
    original_copy[np.where(np_mask_resized == 0)] = 0
    np_original_image[np.where(np_mask_resized != 0)] = 0

    #visual_response = np_heatmap_resized * 0.4 + np_original_image + original_copy * 0.5
    visual_response = np.array(original_image)
    h, w, _ = visual_response.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Stella AI Report'
    scale = 1 * (h / 512)
    thickness = 2
    color_text = (92, 6, 18)
    textsize = cv2.getTextSize(text, font, scale, thickness)[0]

    # Get coords based on boundary
    textX = int((w - textsize[0]) / 2)  # Coord to put the image centered
    textY = int(h - h * 0.03)  # Over 3% of the image height

    cv2.putText(img=visual_response,
                text=text,
                org=(textX, textY),
                fontFace=font,
                fontScale=scale,  # This one scale the image proportionaly to its size (512 is the reference)
                color=color_text,  # Red color
                thickness=thickness)

    text = 'COVID-19'
    scale = scale * 0.5
    thickness = 1
    color_text = (92, 6, 18)
    textsize = cv2.getTextSize(text, font, scale, thickness)[0]

    cv2.putText(img=visual_response,
                text=text,
                org=(round(w * 0.05), round(h - h * 0.1)),
                fontFace=font,
                fontScale=scale,  # This one scale the image proportionaly to its size (512 is the reference)
                color=color_text,  # Red color
                thickness=thickness)

    rectangle_height = h * 0.02

    cv2.rectangle(img=visual_response,
                  #pt1=(round(w * 0.2), round(h - h * 0.1)),
                  pt1=(round(w * 0.07 + textsize[0]), round(h - h * 0.1)),
                  pt2=(round(w - w * 0.05), round(h - h * 0.1 - rectangle_height)),
                  color=(237, 192, 198),
                  thickness=cv2.FILLED)

    cv2.rectangle(img=visual_response,
                  pt1=(round(w * 0.07 + textsize[0]), round(h - h * 0.1)),
                  pt2=(round((w - w * 0.05) * prediction), round(h - h * 0.1 - rectangle_height)),
                  color=(92, 6, 18),
                  thickness=cv2.FILLED)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(round(prediction * 100)) + '%'
    scale = 0.3 * (h / 512)
    thickness = 1
    color_text = (92, 6, 18)
    textsize = cv2.getTextSize(text, font, scale, thickness)[0]

    cv2.putText(img=visual_response,
                text=text,
                org=(round((w - w * 0.05) * prediction + h * 0.01), round(h - h * 0.1 - (rectangle_height - textsize[1])/2)),
                fontFace=font,
                fontScale=scale,  # This one is the scale of the image. It is proportionaly to its size (512 is the reference)
                color=color_text,  # Red color
                thickness=thickness)

    Image.fromarray(np.uint8(visual_response)).convert("RGBA").show()

    cv2.imwrite(output_filename_visual_response, cv2.cvtColor(visual_response, cv2.COLOR_RGB2BGR))

    cv2.destroyAllWindows()

    return output_filename_visual_response
