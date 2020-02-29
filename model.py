from torch import nn
import torch
import numpy as np


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16,3,1,1),
            nn.ELU(),
            nn.BatchNorm2d(16),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1),
            nn.ELU(),
            nn.Conv2d(32,32,3,1,1),
            nn.ELU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(32),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.ELU(),
            nn.Conv2d(64,64,3,1,1),
            nn.ELU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.ELU(),
            nn.Conv2d(128,128,3,1,1),
            nn.ELU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            nn.ELU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ELU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(256,512,3,1,1),
            nn.ELU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ELU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(512)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            nn.ELU(),
            nn.BatchNorm2d(512)
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            nn.ELU(),
            nn.BatchNorm2d(512)
        )
        
        self.deconv91 = nn.Sequential(
            nn.ConvTranspose2d(512,512,3,2,1),
            nn.ConstantPad2d((1,0,1,0),1),
            nn.ELU(),
            nn.BatchNorm2d(512)
        )
        
        self.deconv92 = nn.Sequential(
            nn.Conv2d(768,512,3,1,1),
            nn.ELU(),
            nn.BatchNorm2d(512)
        )
        
        self.deconv101 = nn.Sequential(
            nn.ConvTranspose2d(512,256,3,2,1),
            nn.ConstantPad2d((1,0,1,0),1),
            nn.ELU(),
            nn.BatchNorm2d(256)
        )
        
        self.deconv102 = nn.Sequential(
            nn.Conv2d(384,256,3,1,1),
            nn.ELU(),
            nn.BatchNorm2d(256)
        )
        
        self.deconv111 = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,2,1),
            nn.ConstantPad2d((1,0,1,0),1),
            nn.ELU(),
        )
        
        self.deconv112 = nn.Sequential(
            nn.Conv2d(192,128,3,1,1),
            nn.ELU(),
        )
        
        self.deconv121 = nn.Sequential(
            nn.ConvTranspose2d(128,64,3,2,1),
            nn.ConstantPad2d((1,0,1,0),1),
            nn.ELU(),
        )
        
        self.deconv122 = nn.Sequential(
            nn.Conv2d(96,64,3,1,1),
            nn.ELU(),
        )
        
        self.deconv131 = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,2,1),
            nn.ConstantPad2d((1,0,1,0),1),
            nn.ELU(),
        )
        
        self.deconv132 = nn.Sequential(
            nn.Conv2d(48,32,3,1,1),
            nn.ELU(),
        )

        self.deconv141 = nn.Sequential(
            nn.ConvTranspose2d(32,16,3,1,1),
            nn.ELU(),
        )
        
        self.deconv142 = nn.Sequential(
            nn.Conv2d(16,16,3,1,1),
            nn.ELU(),
        )
        
        self.output = nn.Conv2d(16,3,1,1,0)
        self.output2 = nn.ReLU()
        
    def forward(self,x):  #x is a gray image
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.deconv91(x8)
        x9 = torch.cat([x9, x5], dim=1)
        x9 = self.deconv92(x9)
        x10 = self.deconv101(x9)
        x10 = torch.cat([x10, x4], dim=1)
        x10 = self.deconv102(x10)
        x11 = self.deconv111(x10)
        x11 = torch.cat([x11, x3], dim=1)
        x11 = self.deconv112(x11)
        x12 = self.deconv121(x11)
        x12 = torch.cat([x12, x2], dim=1)
        x12 = self.deconv122(x12)
        x13 = self.deconv131(x12)
        x13 = torch.cat([x13, x1], dim=1)
        x13 = self.deconv132(x13)
        x14 = self.deconv141(x13)
        x14 = self.deconv142(x14)
        y = self.output(x14)
        y = self.output2(y)
        return y
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.bati1 = nn.BatchNorm2d(1)
        self.bato1 = nn.BatchNorm2d(3)
        
        self.convi1 = nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.convo1 = nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.convi2 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.convo2 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.convi3 = nn.Sequential( 
            nn.Conv2d(64,64,3,2,1),
            nn.ReLU()
        )
        self.convo3 = nn.Sequential(
            nn.Conv2d(64,64,3,2,1),
            nn.ReLU()
        )
        
        self.bat2 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,128,3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128,128,3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(128,128,3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(128,256,3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(256,256,3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        self.dp1 = nn.Dropout()
        
        self.fc1 = nn.Sequential( 
            nn.Linear(16384, 100),
            nn.ReLU()
        )
        
        self.dp2 = nn.Dropout()
        
        self.fc2 = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, y):
        x = self.bati1(x)
        y = self.bato1(y)
        
        x = self.convi1(x)
        y = self.convo1(y)
        x = self.convi2(x)
        y = self.convo2(y)
        x = self.convi3(x)
        y = self.convo3(y)
        
        z = torch.cat([x,y], dim=1)
        z = self.bat2(z)
        
        z = self.conv4(z)
        z = self.conv5(z)
        z = self.conv6(z)
        z = self.conv7(z)
        z = self.conv8(z)
        z = z.view(z.size()[0], -1)
        z = self.dp1(z)
        z = self.fc1(z)
        z = self.dp2(z)
        z = self.fc2(z)
        
        return z
    



if __name__ == "__main__":
    img = np.random.rand(1,1,512,512)
    img = torch.tensor(img)
    g = Generator()
    output = g(img.float())
    print(output.size())
    d = Discriminator()
    output2 = d(img.float(), output)
    print(output2)