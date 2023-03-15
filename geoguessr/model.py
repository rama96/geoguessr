from torchvision import models
from torch import nn
from geoguessr.config import OUTPUT_CLASSES , model_name

def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False 
    return model   


class Alexnet:
    def __init__(self):
        self.model = models.alexnet(pretrained=True)
        self.model = freeze_layers(self.model)
    
    def replace_classification_layer(self):
        self.model.classifier[6] = nn.Linear(4096,OUTPUT_CLASSES)
        self.model.classifier.add_module("7",nn.LogSoftmax(dim=1))
    

class VGG16:
    def __init__(self):
        self.model = models.alexnet(pretrained=True)
        self.model = freeze_layers(self.model)

    def replace_classification_layer(self):
        self.model.classifier[6] = nn.Linear(4096,OUTPUT_CLASSES)
        self.model.classifier.add_module("SoftmaxLayer",nn.LogSoftmax(dim=1))


class Resnet18:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model = freeze_layers(self.model)

    def replace_classification_layer(self):
        self.model.fc = nn.Sequential(nn.Linear(512,OUTPUT_CLASSES))
        self.model.fc.add_module('SoftmaxLayer',nn.LogSoftmax(dim=1))
        self.model

class Resnet34:
    def __init__(self):
        self.model = models.resnet34(pretrained=True)
        self.model = freeze_layers(self.model)

    def replace_classification_layer(self):
        self.model.fc = nn.Sequential(nn.Linear(512,OUTPUT_CLASSES))
        self.model.fc.add_module('2',nn.LogSoftmax(dim=1))
        self.model


class Models:
    models = {
            'alexnet':Alexnet(),
            'vgg':VGG16(),
            'resnet_small':Resnet18(),
            'resnet_large':Resnet34(),
        }

    def __init__(self):
        pass        

    @staticmethod
    def prepare_model(model_name = model_name):
        """ Picks the defaul model name from config , also possible to pass into the function """
        model = Models.models[model_name]
        model.replace_classification_layer()
        return model.model

