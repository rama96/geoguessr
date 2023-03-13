from torch.utils.data import Dataset
from dataset import GeoGuessrDataset
from download import  get_images_dataset
from utils import create_dataset_splits

class GeoGuessrDataset(Dataset):
    def __init__(self, ds, label_name, transform=None, target_transform=None):
        
        self.ds = ds
        self.label_name = label_name
        self.transform = transform
        self.target_transform = target_transform
        self.targets = ds[label_name]
        self.classes_to_idx = {j:i for i,j in enumerate(set(ds[label_name]))}
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        
        image = self.ds[idx]['image']
        label = self.ds[idx][self.label_name]
        
        if self.transform:
            image = self.transform(image)        
        
            
        if self.target_transform:
            label = self.target_transform(label)
        
        # Changing label to int using label mapper
        label = self.classes_to_idx[label]
        return image, label
        

def prepare_data():
    
    data = get_images_dataset()    
    
    # X and y transforms
    transforms = transforms.Compose( [transforms.ToTensor()])
    target_transform = None 
    
    geoguessr_dataset = GeoGuessrDataset(ds = data, label_name = 'country_code_iso', transform=transforms, target_transform=target_transform)
    train_dl , test_dl = create_dataset_splits(ds=geoguessr_dataset,TEST_SIZE=0.2,BATCH_SIZE=64)
    
    return train_dl , test_dl
