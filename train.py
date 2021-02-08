# from mini_imagenet_dataloader import MiniImageNetDataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torchmeta.transforms import ClassSplitter, Categorical
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets.miniimagenet import MiniImagenet
from collections import OrderedDict
import torch.nn.functional as F
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from torchmeta.utils.gradient_based import gradient_update_parameters

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
def train():
    transform = transforms.Compose([
        transforms.Resize(84),
        transforms.ToTensor()
    ])
    dataset_transform = ClassSplitter(shuffle=True, num_train_per_class=5, num_test_per_class=5)
    dataset = MiniImagenet('data', transform=transform, num_classes_per_task=5, target_transform=Categorical(num_classes=5) ,meta_split="train", dataset_transform=dataset_transform )

    dataloader = BatchMetaDataLoader(dataset, batch_size=25, shuffle=True)

    model = ModelConvMiniImagenet(5)
    model.to(device='cpu')
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # for batch_idx, batch in enumerate(dataloader):
    #     print(batch["train"][0][0][0].numpy().shape)
    #     plt.imshow(np.rollaxis(batch["train"][0][0][0].numpy(), 0, 3))
    #     plt.show()
    #     break

    accuracy_l = list()

    with tqdm(dataloader, total=15000) as pbar:
        for batch_idx, batch in enumerate(pbar):

            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device='cpu')
            train_targets = train_targets.to(device='cpu')
            # print(train_inputs.shape)
            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device='cpu')
            test_targets = test_targets.to(device='cpu')

            outer_loss = torch.tensor(0., device='cpu')
            accuracy = torch.tensor(0., device='cpu')
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):
                train_logit = model(train_input) 
                inner_loss = F.cross_entropy(train_logit, train_target)
                model.zero_grad()
                params = gradient_update_parameters(model, inner_loss)

                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)
            outer_loss.div_(1)
            accuracy.div_(1)

            outer_loss.backward()
            meta_optimizer.step()
            accuracy_l.append(accuracy.item())
            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
            if(batch_idx >= 15000):
                break

    # filename = os.path.join('', 'maml_omniglot_'
    #     '{0}shot_{1}way.th'.format(5, 5))
    # with open(filename, 'wb') as f:
    #     state_dict = model.state_dict()
    #     torch.save(state_dict, f)
    # plt.plot(accuracy_l[::150])
    # plt.show()

    

if __name__ == "__main__":
    train()