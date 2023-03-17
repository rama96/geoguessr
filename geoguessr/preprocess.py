from geoguessr.download import  get_images_dataset
from geoguessr.utils import create_dataset_splits
from torchvision import transforms
from geoguessr.dataset import GeoGuessrDataset
from geoguessr.config import target_label_name

def prepare_data():
        
    data = get_images_dataset()

    # X and y transforms
    transform = transforms.Compose( 
                [            transforms.Resize((256,256)),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                ]
        )
    target_transform = None 
    
    geoguessr_dataset = GeoGuessrDataset(ds = data, label_name = target_label_name, transform=transform, target_transform=target_transform)
    train_dl , test_dl , train_data_size , test_data_size = create_dataset_splits(ds=geoguessr_dataset)
    
    return train_dl , test_dl , train_data_size , test_data_size  , geoguessr_dataset
