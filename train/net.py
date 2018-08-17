import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0))

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, stride=1, padding=0, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, stride=1, padding=0),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, stride=1, padding=0, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, stride=1, padding=0, groups=2))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        return conv5

class Siamfc(nn.Module):
    def __init__(self, branch):
        super(Siamfc, self).__init__()
        self.branch = branch
        self.bn_adjust = nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def Xcorr(self, x, z):         # x denote search, z denote template
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i, :, :, :].unsqueeze(0), z[i, :, :, :].unsqueeze(0)))
        return torch.cat(out, dim=0)

    def forward(self, x, z):        # x denote search, z denote template
        x = self.branch(x)
        z = self.branch(z)
        xcorr_out = self.Xcorr(x, z)
        out = self.bn_adjust(xcorr_out)
        return out

    def load_params(self, net_path):
        checkpoint = torch.load(net_path)
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
            self.load_state_dict(state_dict)

    def load_params_from_mat(self, net_path):
        params_names_list, params_values_list = load_matconvnet(net_path)
        params_values_list = [torch.from_numpy(p) for p in params_values_list]  # values convert numpy to Tensor

        for index, param in enumerate(params_values_list):
            param_name = params_names_list[index]
            if 'conv' in param_name and param_name[-1] == 'f':
                param = param.permute(3, 2, 0, 1)
            param = torch.squeeze(param)
            params_values_list[index] = param

        net = nn.Sequential(
            self.branch.conv1,
            self.branch.conv2,
            self.branch.conv3,
            self.branch.conv4,
            self.branch.conv5)

        for index, layer in enumerate(net):
            layer[0].weight.data[:] = params_values_list[params_names_list.index('conv%df' % (index+1))]
            layer[0].bias.data[:] = params_values_list[params_names_list.index('conv%db' % (index+1))]

            if index < len(net) - 1:
                layer[1].weight.data[:] = params_values_list[params_names_list.index('bn%dm' % (index+1))]
                layer[1].bias.data[:] = params_values_list[params_names_list.index('bn%db' % (index+1))]
                bn_moments = params_values_list[params_names_list.index('bn%dx' % (index+1))]
                layer[1].running_mean[:] = bn_moments[:, 0]
                layer[1].running_var[:] = bn_moments[:, 1] ** 2
            else:
                self.bn_adjust.weight.data[:] = params_values_list[params_names_list.index('adjust_f')]
                self.bn_adjust.bias.data[:] = params_values_list[params_names_list.index('adjust_b')]


def load_matconvnet(net_path):
    mat = scipy.io.loadmat(net_path)
    net_dot_mat = mat.get('net')         # get net
    params = net_dot_mat['params']      # get net/params
    params = params[0][0]
    params_names = params['name'][0]    # get net/params/name
    params_names_list = [params_names[p][0] for p in range(params_names.size)]
    params_values = params['value'][0]  # get net/params/val
    params_values_list = [params_values[p] for p in range(params_values.size)]

    return params_names_list, params_values_list


# Test Code
if __name__ == '__main__':
    # test AlexNet
    template = torch.randn(3, 3, 127, 127)
    search = torch.randn(3, 3, 255, 255)
    alexnet = AlexNet()
    print(alexnet(template).size())
    print(alexnet(search).size())

    # test Siamfc
    siamfc = Siamfc(branch=AlexNet())
    print(siamfc(search, template).size())