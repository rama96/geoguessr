
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# generate indices: instead of the actual data we pass in integers instead


def create_dataset_splits(ds,TEST_SIZE = 0.2 , BATCH_SIZE = 64):
    """ Stratifies the dataset according to the label to sample it into train and test """
    SEED = 42

    train_indices, test_indices, _, _ = train_test_split(
    range(len(ds)),
    ds.targets,
    stratify=ds.targets,
    test_size=TEST_SIZE,
    random_state=SEED
    )

    # generate subset based on indices
    train_split = Subset(ds, train_indices)
    test_split = Subset(ds, test_indices)

    # create batches
    train_dl = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_split, batch_size=BATCH_SIZE)

    return train_dl , test_dl

    
    