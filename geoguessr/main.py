from geoguessr.preprocess import prepare_data
from geoguessr.model import Models
from geoguessr.config import model_name , experiment_name , EPOCHS
from geoguessr.train import train_and_validate
import torch.optim as optim
from torch import nn
from geoguessr import DIR_PLOTS , DIR_HISTORY , DIR_MODEL
import torch

def main():
    
    print(" .. Preparing_data")
    train_dl , test_dl , train_data_size , test_data_size  , geoguessr_dataset =  prepare_data()
    model = Models.prepare_model(model_name)
    
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())
    
    print(" .. Train and Validating ")
    trained_model , history = train_and_validate(model, loss_func, optimizer, train_dl , test_dl , train_data_size , test_data_size , epochs=EPOCHS)

    print(" .. Saving the model ")
    model_filepath = DIR_MODEL.joinpath(f'geoguessr_{model_name}_{experiment_name}.pth')
    torch.save(trained_model.state_dict(), model_filepath)

    print(" .. Saving the History ")
    history_path = DIR_HISTORY.joinpath(f'{model_name}__{experiment_name}_history.pt')
    torch.save(history, history_path)

    import numpy as np
    import matplotlib.pyplot as plt
    
    print(" .. Plotting Loss curves ")
    history = np.array(history)
    plt.plot(history[:,0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0,10)
    loss_filepath = DIR_PLOTS.joinpath(f'{model_name}_{experiment_name}_loss_curve.png')
    plt.savefig(loss_filepath)
    plt.show()    

    print(" .. Plotting Accuracy curves ")
    plt.plot(history[:,2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    acc_filepath = DIR_PLOTS.joinpath(f'{model_name}_{experiment_name}_accuracy_curve.png')
    plt.savefig(acc_filepath)
    plt.show()




if __name__ == "__main__" :
    main()
