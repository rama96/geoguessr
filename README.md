# geoguessr
A n application to help with the game geoguessr

# Downloading dataset
Download kaggle creds from kaggle and store it in ~/.kaggle/kaggle.json

```bash
kaggle datasets download -d ubitquitin/geolocation-geoguessr-images-50k
```

#env setup
printf "\n# Adding this command to read local .env file" >> env/bin/activate
printf "\nexport \$(grep -v '^#' .env | xargs)" >> env/bin/activate
