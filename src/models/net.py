import torchvision.models as models
import torch.nn as nn
import timm
import torch

def choose_backbone(backbone: str, pretrained: bool = True, drop_rate: float = 0.2):
    if backbone == "resnet101":
        return models.resnet101(pretrained=pretrained)
    if backbone == "resnet152":
        return models.resnet152(pretrained=pretrained)

    return timm.create_model(backbone, pretrained=pretrained, drop_rate=drop_rate)

class Net(nn.Module):
    def __init__ (self, config):

        super(Net, self).__init__()
        self.config = config
        self.num_classes = self.cfg.network['num_classes']
        self.embedding_size = self.cfg.network['embedding_size']
        self.freeze_params = self.cfg.network['freeze_params']
        self.drop_rate = self.cfg.network['drop_rate']
        self.backbone = self.cfg.network['backbone']


        assert "efficientnet" in self.backbone, "Haven't develop on other backbone, please use effcientnet for now."
        self.root_model = choose_backbone(self.backbone, pretrained=True, drop_rate=self.drop_rate)

        if "efficientnet" in self.backbone:
            self.backbone_features = self.root_model.classifier.in_features  # (N, 1280, 7, 7) -> (N, 1280)

            tmp = list(self.root_model.children())
            layers = tmp[:2] + list(tmp[2].children())[:5]
            self.back_layers = nn.Sequential(*layers)

            layers = list(tmp[2].children())[5:] + list(tmp)[3:-1]
            head_layers = nn.Sequential(*layers)

            if self.freeze_params["back_layer"]:
                for param in self.back_layers.parameters():
                    param.requires_grad = False
            if self.freeze_params["head_layer"]:
                for param in head_layers.parameters():
                    param.requires_grad = False

            fc = nn.Sequential(
                nn.Linear(self.backbone_features, self.embedding_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(num_features=self.embedding_size),
                # nn.Dropout(self.drop_rate),
            )

            self.classifier_final = nn.Sequential(*[head_layers, fc, nn.Linear(self.embedding_size, self.num_classes)])


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (N, 3, 360, 360)

        Returns:
        """

        x = self.back_layers(x)
        pred = self.classifier_final(x)


        return pred



def build_model(num_classes):
    model = models.efficientnet_b0(pretrained=True)
    # model = models.resnet18(pretrained=True)

    for params in model.parameters():
        params.requires_grad = False

    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model

def alex_model(num_classes):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model.classifier[4] = nn.Linear(4096,100)
    model.classifier[6] = nn.Linear(100,num_classes)
    return model


