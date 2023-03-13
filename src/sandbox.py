

from download import  get_images_dataset
from torchvision import transforms


if __name__ == "__main__":

    train_data = get_images_dataset()
    
    for i in range(5):
        
        img = train_data[i]['image']

        transform  = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ])
        
        img_t = transform(img)
        fin_img = transforms.ToPILImage()(img_t.squeeze_(0))
        fin_img.show()
