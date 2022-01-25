from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_fair_7 = torchvision.models.resnet34(pretrained=True)
model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
# model_fair_7.load_state_dict(torch.load('./models/fairface_alldata_20191111.pt'))
model_fair_7.load_state_dict(torch.load('./models/res34_fair_align_multi_7_20190809.pt'))
model_fair_7 = model_fair_7.to(device)
model_fair_7.eval()

# model_fair_4 = torchvision.models.resnet34(pretrained=True)
# model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
# model_fair_4.load_state_dict(torch.load(
#     './models/fairface_alldata_4race_20191111.pt'))
# model_fair_4 = model_fair_4.to(device)
# model_fair_4.eval()


def fair_iterator(image_batch):
    # TODO: Allow choice of 4 class race classifier
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    for image in image_batch:
        # Should already be normalized
        image = trans(image)

        # TODO: fairface uses double our previous
        # chunk size, should adapt this
        # TODO: Check whether these methods are only
        # available on dlib images
        image = image.view(1, 3, 224, 224)
        image = image.to(device)

        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)
        yield ('fair7', age_pred, gender_pred, race_pred)
