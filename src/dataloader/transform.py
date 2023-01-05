import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms


class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, image, *args, **kwargs):
        return self.transforms(image=np.array(image))['image']


def transform_val(method: str,
                mean: list[float] = [0.485, 0.456, 0.406],
                std: list[float] = [0.229, 0.224, 0.225]):
    val_transform = Transforms(A.Compose(
        [   
            A.CenterCrop(width=640, height=640, p = 0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    ))

    return val_transform


def transform_train(method: str,
                mean: list[float] = [0.485, 0.456, 0.406],
                std: list[float] = [0.229, 0.224, 0.225],
                p: float = 0.5):
    

    train_transform = Transforms(A.Compose(
        [   
            A.CenterCrop(width=640, height=640, p = 0.5),
            A.RandomRotate90(p=p),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    ))
    return train_transform

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
