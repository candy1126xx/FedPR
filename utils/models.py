import os, random
import torch, torch.nn as nn
import torch.nn.functional as F


def create_model(model_name):
    if model_name=='mnistcnn4':
        output_channel = random.choice([60, 64])
        model = MnistCNN4(output_channel)
        model_name = 'mnistcnn4_' + str(output_channel)
    elif model_name=='cifarcnn4':
        model = CifarCNN4()
    elif model_name=='resnet18':
        model = ResNet18()
    return model_name, model


# 分层模型
class LayeringModel(nn.Module):
    def __init__(self):
        super(LayeringModel, self).__init__()
        self.representor = None
        self.predictor = None
        
    def forward(self, x):
        representation = self.representor(x)
        logits = self.predictor(representation)
        scores = F.log_softmax(logits, dim=1)
        return representation, logits, scores

    def forward_rep(self, representation):
        logits = self.predictor(representation)
        scores = F.log_softmax(logits, dim=1)
        return logits, scores

    def freeze_pre(self):
        for param in self.predictor.parameters():
            param.requires_grad = False
        self.predictor.eval()
    
    def freeze_rep(self):
        for param in self.representor.parameters():
            param.requires_grad = False
        self.representor.eval()

    def weight_flatten(self):
        params = []
        for u in self.parameters():
            params.append(u.clone().detach().view(-1))
        params = torch.cat(params)
        return params

    def grad_flatten(self):
        grads = []
        for u in self.parameters():
            if u.requires_grad:
                grads.append(u.grad.clone().detach().view(-1))
        grads = torch.cat(grads)
        return grads


class Linear(LayeringModel):
    def __init__(self):
        super(Linear, self).__init__()
        self.representor = nn.Sequential(
            nn.Flatten(1),
        )
        self.predictor = nn.Linear(in_features=64, out_features=4, bias=True)


class MLP(LayeringModel):
    def __init__(self):
        super(MLP, self).__init__()
        self.representor = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(64, 16),
        )
        self.predictor = nn.Linear(in_features=16, out_features=4, bias=True)


class MnistMLP(LayeringModel):
    def __init__(self):
        super(MnistMLP, self).__init__()
        self.representor = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )
        self.predictor = nn.Linear(in_features=32, out_features=10, bias=True)


class MnistCNN2(LayeringModel):
    def __init__(self):
        super(MnistCNN2, self).__init__()
        self.representor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(784, 32),
        )
        self.predictor = nn.Linear(in_features=32, out_features=10, bias=False)


class MnistCNN4(LayeringModel):
    def __init__(self, output_channel):
        super(MnistCNN4, self).__init__()
        if output_channel==60:
            channel = 240
        elif output_channel==64:
            channel = 256
        self.representor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, output_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(channel, 32),
        )
        self.predictor = nn.Linear(in_features=32, out_features=10, bias=False)


class CifarCNN4(LayeringModel):
    def __init__(self):
        super(CifarCNN4, self).__init__()
        self.representor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Flatten(1),
        )
        self.predictor = nn.Linear(in_features=256, out_features=10, bias=True)



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(LayeringModel):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.representor = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride = 2, padding=1),
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]),
            self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]),
            self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(512, 256),
        )
        self.predictor = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)


def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def ResNet18(pretrained=False, progress=True, device="cpu", **kwargs):
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, **kwargs
    )
