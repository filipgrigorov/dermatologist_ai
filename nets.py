import torch
import torch.nn as nn

import torchvision.models as models

def print_layer_sizes(net):
    for child in net.children():
        for op in child.parameters():
            if op.requires_grad:
                print('{}: {}'.format(type(op), op.size()))

def print_learnable_parameters(net):
    count = 0
    for child in net.children():
        for op in child.parameters():
            if op.requires_grad:
                count += op.numel()
    print('Learnable parameters: ', count)

class DermatologistNet(nn.Module):
    def __init__(self, noutputs, is_debug=False):
        super(DermatologistNet, self).__init__()
        self.inceptionv3 = models.inception_v3(pretrained=True, progress=True)

        num_feats = self.inceptionv3.AuxLogits.fc.in_features
        self.inceptionv3.AuxLogits.fc = nn.Linear(num_feats, noutputs)

        num_feats = self.inceptionv3.fc.in_features
        self.inceptionv3.fc = nn.Linear(num_feats, noutputs)

        self.input_size = 299

        if is_debug:
            print(self.inceptionv3)

    def forward(self, x):
        return self.inceptionv3(x)

if __name__ == '__main__':
    net = DermatologistNet(noutputs=3)

    dummy = torch.zeros((2, 3, 299, 299))
    out = net(dummy)
    print('\noutputs (logits): ', out.logits)
    print('\noutputs (auxilliary logits): ', out.aux_logits)

    if __debug__:
        # print_layer_sizes(net)
        print_learnable_parameters(net)

