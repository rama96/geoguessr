from geoguessr.preprocess import prepare_data


def main():
    
    train_dl , test_dl , train_data_size , test_data_size  , geoguessr_dataset =  prepare_data()
    
    

if __name__ == "__main__" :
    main()
