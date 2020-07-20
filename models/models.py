import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError

class NormalizeLayer():
    def __init__(self, norm='ETR'):
        self.norm = norm
    
    def normalize(self, x):
        if self.norm == 'ABS':
            maxs = torch.max(torch.abs(x), dim=1, keepdim=True).values
            x = torch.div(x, maxs)
        elif self.norm == 'FBN':
            n = torch.norm(x, dim=1, keepdim=True) + 1e-8
            x = torch.div(x, n)
        else: 
            w,h = x.shape
            lastEntries = torch.reshape(x[:,-1], (w,-1)) + 1e-8
            x = torch.div(x, lastEntries)
        return x

class DeepFMatNet(Base):
    def __init__(self, outputSize, norm='ETR'):
        super(DeepFMatNet, self).__init__()
        self.conv1 = nn.Sequential (
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )


        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout(0.37)

        self.fc1 = nn.Linear(256*32*32, 1024)

        self.fc = nn.Sequential(
            self.fc1,
            self.dropout3,
            nn.Linear(1024, 512),
            nn.Linear(512, outputSize)
        )
        
        self.normLayer = NormalizeLayer(norm=norm)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        w,h = x.shape[-1], x.shape[-2]
        x , indices= F.max_pool2d(x,2, return_indices=True)

        indices = indices.float()/ (w*h)
        x = torch.cat((indices,x), 1)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.normLayer.normalize(x)

        return x

# class MyDataParallel

class DeepFMatVGG16(Base):
   def __init__(self, outputSize, norm='ETR'):
       super(DeepFMatVGG16, self).__init__()
       self.vgg = torchvision.models.vgg16(pretrained=True)

       pretrained1stFeatureWeight = self.vgg.features[0].weight
       newFeatures = nn.Sequential(*list(self.vgg.features.children()))
       newFeatures[0] = nn.Conv2d(2, 64, kernel_size=3, padding=1)
       newFeatures[0].weight.data.normal_(0, 0.001)
       newFeatures[0].weight.data = nn.Parameter(pretrained1stFeatureWeight[:, :2, :, :])

       self.vgg.features = newFeatures

       lastClassifierLayerNm = self.vgg.classifier[-1].in_features
       lastClassifierLayer = nn.Linear(lastClassifierLayerNm, outputSize)
       newClassifier = nn.Sequential( *list(self.vgg.classifier.children())[:-1]
                                      ,lastClassifierLayer
                                      )

       self.vgg.classifier = newClassifier
       
       self.normLayer = NormalizeLayer(norm=norm)

   def forward(self, x):
       x = self.vgg(x)
       x = self.normLayer.normalize(x)
       return x

class DeepFMatResNet18(Base):
    def __init__(self, outputSize, norm='ETR'):
        super(DeepFMatResNet18, self).__init__()
        self.resNet = torchvision.models.resnet18(pretrained=True)

        pretrained1stConvdWeight = self.resNet.conv1.weight
        newConv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        newConv1.weight.data.normal_(0, 0.001)
        newConv1.weight.data = nn.Parameter(pretrained1stConvdWeight[:, :2, :, :])

        self.resNet.conv1 = newConv1

        lastClassifierLayerNm = self.resNet.fc.in_features
        newFc = nn.Linear(in_features=lastClassifierLayerNm, out_features=outputSize)

        self.resNet.fc = newFc
        self.normLayer = NormalizeLayer(norm=norm)

    def forward(self, x):
        x = self.resNet(x)
        x = self.normLayer.normalize(x)
        return x

class DeepFMatAlex(Base):
    def __init__(self, outputSize, norm='ETR'):
        super(DeepFMatAlex, self).__init__()
        self.alex = torchvision.models.alexnet(pretrained=True)

        pretrained1stFeatureWeight = self.alex.features[0].weight
        newFeatures = nn.Sequential(*list(self.alex.features.children()))
        newFeatures[0] = nn.Conv2d(2, 64, kernel_size=11, stride=4, padding=2)
        newFeatures[0].weight.data.normal_(0, 0.001)
        newFeatures[0].weight.data = nn.Parameter(pretrained1stFeatureWeight[:, :2, :, :])

        self.alex.features = newFeatures

        lastClassifierLayerNm = self.alex.classifier[-1].in_features
        lastClassifierLayer = nn.Linear(lastClassifierLayerNm, outputSize)
        newClassifier = nn.Sequential(*list(self.alex.classifier.children())[:-1],
                                      lastClassifierLayer)
        self.alex.classifier = newClassifier
        self.normLayer = NormalizeLayer(norm=norm)



    def forward(self, x):
        x = self.alex(x)
        x = self.normLayer.normalize(x)
        return x



    