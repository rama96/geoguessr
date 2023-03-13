# geoguessr
A n application to help with the game geoguessr

#env setup
printf "\n# Adding this command to read local .env file" >> env/bin/activate
printf "\nexport \$(grep -v '^#' .env | xargs)" >> env/bin/activate
