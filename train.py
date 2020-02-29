import os
import cv2
import numpy as np
from tqdm import tqdm # Displays a progress bar
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from model import *
import random


# constants
epochs = 50
x_shape = 512
y_shape = 512
fixed_seed_num = 1234
np.random.seed(fixed_seed_num)


device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
gen = Generator().to(device)
disc = Discriminator().to(device)

criterion_discriminator = nn.BCELoss() # Specify the loss layer
cgan_loss_weight = [5,100]
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer_disc = optim.Adam(disc.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-08) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
optimizer_cGAN = optim.Adam(gen.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-08)


# constants
dataset = 'dataset/train/'
store2 = 'dataset/generated_images/'
val_data = 'dataset/validation/'
store = 'dataset/generated_Images/'


samples = len(os.listdir(dataset))
val_samples = len(os.listdir(val_data))
rgb = np.zeros((samples, x_shape, y_shape, 3))
gray = np.zeros((samples, x_shape, y_shape, 1))
rgb_val = np.zeros((val_samples, x_shape, y_shape, 3))
gray_val = np.zeros((val_samples, x_shape, y_shape, 1))

'''
for i, image in enumerate(os.listdir(dataset)[:samples]):
    I = cv2.imread(dataset+image)
    I = cv2.resize(I, (x_shape, y_shape))
#     J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
#     J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb[i] = I;
#     gray[i] = J

for i, image in enumerate(os.listdir(val_data)[:val_samples]):
    I = cv2.imread(val_data+image)
    I = cv2.resize(I, (x_shape, y_shape))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb_val[i] = I; gray_val[i] = J
    


data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
manga_dataset = datasets.ImageFolder(root=dataset, transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(manga_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)

for data in dataset_loader:
    print(data.size())
'''

# trains cGAN model
def train(gen, disc, disc_trainloader,cGAN_trainloader, valloader):
    
    print("Start training...")
    
    samples = len(rgb)


    #gray = gray.to(device)
    #optimizer.zero_grad()
    
    criterion = nn.BCELoss()
    def cos(y_pred, y_true):
        return torch.sum(y_pred*y_true)/torch.norm(y_pred)/torch.norm(y_true)
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    # todo: split into 4 battches
    #disc.fit([inputs, outputs], y, batch_size=4)
    batch_size = 4
    num_epoch = 1
    disc.train()
    gen.eval()
    for i in range(num_epoch):
        running_loss = []
        for grays, true_rgbs in tqdm(disc_trainloader):
            print(grays.shape)
            grays = grays.to(device)
            true_rgbs = true_rgbs.to(device)
            grays = grays.permute((0,3,1,2))
            true_rgbs = true_rgbs.permute((0,3,1,2))
            generated_rgbs = gen(grays.float())
            optimizer_disc.zero_grad() # Clear gradients from the previous iteration
            pred1 = disc(grays.float(), true_rgbs.float())
            pred2 = disc(grays.float(), generated_rgbs.float())
            label1 = torch.ones([grays.size()[0], 1], dtype=torch.float, device=device)
            label2 = torch.zeros([grays.size()[0], 1], dtype=torch.float, device=device)
            loss1 = criterion(pred1, label1) # Calculate the loss
            loss2 = criterion(pred2, label2)
            loss = loss1 + loss2
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer_disc.step() # Update trainable weights
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch


    # # cGAN.fit(gray, [np.ones((samples, 1)), rgb], epochs=1, batch_size=batch, callbacks=[tensorboard],validation_data=[gray_val,[np.ones((val_samples,1)),rgb_val]])
    # cGAN.fit(gray, [np.ones((samples, 1)), rgb], batch_size=batch,
    #          validation_data=[gray_val, [np.ones((val_samples, 1)), rgb_val]])
    # todo: split into  cGAN_trainloader battches
    num_epoch = 1
    disc.eval()
    gen.train()
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(cGAN_trainloader):
            batch = batch.to(device).float()
            label = label.to(device).float()
            #plt.figure()
            #plt.imshow(label[0].cpu().numpy().astype(int))
            batch = batch.permute((0,3,1,2))
            label = label.permute((0,3,1,2))
            optimizer_cGAN.zero_grad() # Clear gradients from the previous iteration
            gen_image = gen(batch).float()
            #plt.figure()
            #plt.imshow(gen_image[0].permute(1,2,0).detach().cpu().numpy().astype(int))
            y = disc(batch, gen_image)
            y_truth = torch.ones([batch.size()[0], 1], dtype=torch.float, device=device)
            loss =  5*criterion(y, y_truth) # Calculate the loss
            loss += 100*(mse(gen_image, label)+(1+cos(gen_image,label))*mae(gen_image, label))
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer_cGAN.step() # Update trainable weights
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
        #train_loss_history.append(np.mean(running_loss))
        '''
        #print("Evaluate on validation set...")
        #correct = 0
        running_loss_val = []
        #TODO SPLIT INTO VALLOADER BATCHES
        disc.eval()
        gen.eval()
        for batchval, labelval in tqdm(valloader):
            batchval = batchval.to(device)
            labelval = labelval.to(device)
            batchval = batchval.permute((0,3,1,2))
            labelval = labelval.permute((0,3,1,2))
            gen_image_val = gen(batchval.float())
            y_val1 = disc(batchval.float(), gen_image_val.float())
            y_val2 = disc(batchval.float(), labelval.float())
            y_val_truth1 = torch.zeros([batch_size, 1], dtype=torch.float, device=device)
            y_val_truth2 = torch.ones([batch_size, 1], dtype=torch.float, device=device)
            loss_val = cgan_loss_weight[0] * (criterion(y_val1, y_val_truth1)) # Calculate the loss
            loss_val += cgan_loss_weight[0] * (criterion(y_val2, y_val_truth2))
            loss_val += cgan_loss_weight[1] * custom_loss_2(labelval.float(), gen_image_val.float())
            running_loss_val.append(loss_val.item())
            

            #correct += (torch.argmax(predval, dim=1) == labelval).sum().item()
        #val_loss_history.append(np.mean(running_loss_val))
        #acc = correct / (len(valloader.dataset))
        #print("Evaluation accuracy: {}".format(acc))
        '''

# get train dataset
y_train = np.zeros((samples, 1))
for i, image in enumerate(os.listdir(dataset)[:samples]):
    I = cv2.imread(dataset + image)
    I = cv2.resize(I, (x_shape, y_shape))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb[i] = I;
    gray[i] = J

# get validation dataset
for i, image in enumerate(os.listdir(val_data)[:val_samples]):
    I = cv2.imread(val_data + image)
    I = cv2.resize(I, (x_shape, y_shape))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb_val[i] = I;
    gray_val[i] = J

#todo
# initialise data generator
'''
datagen = ImageDataGenerator(zoom_range=0.2, fill_mode='wrap', horizontal_flip=True,
                             vertical_flip=True, rotation_range=15)
datagen.fit(rgb)
'''

        
class MangaDataset(Dataset):
    def __init__(self, rgb, gray, transform=None):
        self.rgb = rgb
        self.gray = gray
        self.transform = transform
    
    def __len__(self):
        return len(self.rgb)
    
    def __getitem__(self,idx):
        image_left = self.gray[idx]
        image_right = self.rgb[idx]
        # Random horizontal flipping
        if random.random() > 0.5:
            image_left = np.flip(image_left,1)
            image_right = np.flip(image_right,1)
        # Random vertical flipping
        if random.random() > 0.5:
            image_left = np.flip(image_left,2)
            image_right = np.flip(image_right,2)
        return (torch.tensor(self.gray[idx]), torch.tensor(self.rgb[idx]))

disc_trainloader = DataLoader(MangaDataset(rgb, gray), batch_size=4)
cGAN_trainloader = DataLoader(MangaDataset(rgb, gray), batch_size=4)
valloader = DataLoader(MangaDataset(rgb, gray), batch_size=4)

        
epochs = 200
b=1

try:
  gen.load_state_dict(torch.load('gen.pt'))
except Exception as e:
  print(e)
try:
  disc.load_state_dict(torch.load('disc.pt'))
except Exception as e:
  print(e)

for e in range(epochs):
    print('Epoch', e)
    train(gen,disc,disc_trainloader,cGAN_trainloader, valloader)
    gray_val_tensor = torch.tensor(gray_val)
    gray_val_tensor = gray_val_tensor.permute((0,3,1,2))
    gray_val_tensor.to(device)
    gen.eval()
    gen_image_val = gen(gray_val_tensor.float().to(device))
    gen_image_val = gen_image_val.permute((0,2,3,1))
    if e%1 == 0: 
        for j in range(val_samples):
            if not os.path.exists(store2):
                os.mkdir(store2)
            output = gen_image_val[j].detach().cpu().numpy()
            output = (output-np.min(output))/(np.max(output)-np.min(output))*255
            cv2.imwrite(store2+str(e)+str(j)+'.jpg', output)
    if e%5 == 0:
        torch.save(gen.state_dict(), 'gen.pt')
        torch.save(disc.state_dict(), 'disc.pt')