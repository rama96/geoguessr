from torch.utils.data import Dataset

class GeoGuessrDataset(Dataset):
    def __init__(self, ds, label_name, transform=None, target_transform=None):
        
        self.ds = ds
        self.label_name = label_name
        self.transform = transform
        self.target_transform = target_transform
        self.targets = ds[label_name]
        self.classes_to_idx = {j:i for i,j in enumerate(set(ds[label_name]))}
        self.idx_to_classes = {j:i for i,j in self.classes_to_idx.items()}
        
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
        

