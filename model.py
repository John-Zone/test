from torchvision.models import resnet34
import numpy as np
import torch


def get_model():
    model = resnet34(pretrained=True)
    num_fc_ftr = model.fc.in_features
    model.fc = torch.nn.Linear(num_fc_ftr, 2)
    return model
