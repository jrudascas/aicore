import os
from os.path import join
from torchvision import transforms


TRANSFORMER = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

COVID19_CT_PATH_WEIGHTS = join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/covid_denseNet169_ep_100_acc_1.pt')
COVID19_CT_PATH_SAVE_VISUAL_RESPONSE = '/tmp/'