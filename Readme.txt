This is the project for colorizing Manga images using cGAN. It's implemented with Pytorch.
The model.py contains all the model architecture for Generator and Discriminator.
The train.py can be run to train the model.
All the image data are in the dataset folder, the training data are in the dataset/train, validation data are in the dataset/validate.
We have 118 512x512 images for training, and two 512x512 images for validation.
After run the train.py, a new folder dataset/generated_images will be created, which includes the colorized validation image by Generator for each epoch. 