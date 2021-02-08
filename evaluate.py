# from mini_imagenet_dataloader import MiniImageNetDataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torchmeta.transforms import ClassSplitter, Categorical
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets.miniimagenet import MiniImagenet
from collections import OrderedDict
import torch.nn.functional as F
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from torchmeta.utils.gradient_based import gradient_update_parameters
from skimage import io, transform


def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
            track_running_stats=False)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))


def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())



class OwnDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, train=True):
            'Initialization'
            self.train = train
            if(train == True):
                self.data_train = []
                self.data_train_labels = []
                for i in range(5):
                    for j in range(5):
                        self.data_train.append(io.imread('data/evaluation/train/'+str(i)+'/'+str(j)+'.JPEG') / 255)
                        self.data_train_labels.append(i)
                for i in range(25):
                    self.data_train[i] = self.transform_img(self.data_train[i])
                self.data_train = torch.tensor(self.data_train, dtype=torch.float)
                self.data_train_labels = torch.tensor(self.data_train_labels, dtype=torch.long)
            else:
                self.data_test = []
                self.data_test_labels = []
                for i in range(5):
                    for j in range(5):
                        self.data_test.append(io.imread('data/evaluation/test/'+str(i)+'/'+str(j)+'.JPEG') / 255)
                        self.data_test_labels.append(i)
                for i in range(25):
                    self.data_test[i] = self.transform_img(self.data_test[i])
                self.data_test = torch.tensor(self.data_test)
                self.data_test_labels = torch.tensor(self.data_test_labels)
            self.list_IDs = range(25)

    def __len__(self):
            'Denotes the total number of samples'
            return 25
    
    def __getitem__(self, index):
        if(self.train == True):
            sample = {'image': self.data_train[index].permute(2, 0, 1), 'label': self.data_train_labels[index]}
        else:
            sample = {'image': self.data_test[index].permute(2, 0, 1), 'label': self.data_test_labels[index]}
        return sample

    def transform_img(self, img):
        return transform.resize(img, (84,84))
    

class MetaConvModel(MetaModule):
    """4-layer Convolutional Neural Network architecture from [1].
    Parameters

    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True))
        ]))
        self.classifier = MetaLinear(feature_size, out_features, bias=True)
        
    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits


def ModelConvMiniImagenet(out_features, hidden_size=84):
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=5 * 5 * hidden_size)

if __name__ == "__main__":
    model = ModelConvMiniImagenet(5)
    model.load_state_dict(torch.load('maml_omniglot_5shot_5way.th'))
    model.zero_grad()

    data_train = OwnDataset(train=True)
    data_test = OwnDataset(train=False)

    trainloader = DataLoader(data_train, batch_size=25)
    testloader = DataLoader(data_test, batch_size=25)

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    accuracy_l = list()
    loss = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(4):
        for batch_idx, batch in enumerate(trainloader):
            
            model.zero_grad()
            train_inputs = batch['image'].float()
            train_targets = batch['label'].long()
        
            outer_loss = torch.tensor(0.)
            accuracy = torch.tensor(0.)
            train_logit = model(train_inputs)
            inner_loss = F.cross_entropy(train_logit, train_targets)
            model.zero_grad()   
            params = gradient_update_parameters(model, inner_loss)

            test_logit = model(train_inputs , params=params)
            outer_loss += F.cross_entropy(test_logit, train_targets)

            with torch.no_grad():
                accuracy += get_accuracy(test_logit, train_targets)
            outer_loss.div_(1)
            accuracy.div_(1)

            outer_loss.backward()
            meta_optimizer.step()
            accuracy_l.append(accuracy.item())
    plt.plot(accuracy_l)
    plt.show()
    accuracy_l.clear()

    # model.eval()
    # filename = os.path.join('', 'maml_omniglot_eval_'
    #     '{0}shot_{1}way.th'.format(5, 5))
    # with open(filename, 'wb') as f:
    #     state_dict = model.state_dict()
    #     torch.save(state_dict, f)
    # plt.plot(accuracy_l)
    # plt.show()