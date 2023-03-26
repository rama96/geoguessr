from torch.utils.data import Dataset
from fastai.vision import * 


path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_folder(path, )