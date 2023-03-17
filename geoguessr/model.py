from torchvision import models
import timm
from torch import nn
from geoguessr.config import OUTPUT_CLASSES , model_name

def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False 
    return model   


class Alexnet:
    def __init__(self, from_timm = False):
        if from_timm:
            self.model = timm.create_model('resnet18',pretrained=True)
        else : 
            self.model = models.alexnet(pretrained=True)
        self.model = freeze_layers(self.model)
    
    def replace_classification_layer(self):
        self.model.classifier[6] = nn.Linear(4096,OUTPUT_CLASSES)
        self.model.classifier.add_module("7",nn.LogSoftmax(dim=1))
    

class VGG16:
    def __init__(self, from_timm = False):
        if from_timm:
            self.model = timm.create_model('vgg16',pretrained=True)
        else :
            self.model = models.vgg16(pretrained=True)
        self.model = freeze_layers(self.model)

    def replace_classification_layer(self):
        self.model.classifier[6] = nn.Linear(4096,OUTPUT_CLASSES)
        self.model.classifier.add_module("SoftmaxLayer",nn.LogSoftmax(dim=1))


class Resnet18:
    def __init__(self, from_timm = False):
        if from_timm:
            self.model = timm.create_model('resnet18',pretrained=True)
        else :
            self.model = models.resnet18(pretrained=True)
        self.model = freeze_layers(self.model)

    def replace_classification_layer(self):
        self.model.fc = nn.Sequential(nn.Linear(512,OUTPUT_CLASSES))
        self.model.fc.add_module('SoftmaxLayer',nn.LogSoftmax(dim=1))
        self.model

class Resnet34:
    def __init__(self, from_timm = False):
        if from_timm:
            self.model = timm.create_model('resnet34',pretrained=True)
        else :
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

    timm_models = {
            'alexnet':Alexnet(True),
            'vgg':VGG16(True),
            'resnet_small':Resnet18(True),
            'resnet_large':Resnet34(True),
        }

    def __init__(self):
        pass        

    @staticmethod
    def prepare_model(model_name = model_name , from_timm = False):
        """ Picks the defaul model name from config , also possible to pass into the function """
        if from_timm :
            model = Models.timm_models[model_name]
        else :
            model = Models.models[model_name]
        model.replace_classification_layer()
        return model.model

