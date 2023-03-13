from datasets import load_dataset , list_datasets

dataset_url = "'stochastic/random_streetview_images_pano_v0.0.2'"

def get_images_dataset():
    datasets = load_dataset(dataset_url)
    train_data = datasets['train']
    return train_data