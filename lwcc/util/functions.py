import os
import gdown
import numpy as np
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision import transforms


img_type = Union[str, os.PathLike, np.ndarray, Image.Image]


def tensor_convert(tensor):
    if tensor.device != torch.device("cpu"):
        tensor = tensor.cpu()
    return tensor.numpy()


def build_url(path):
    url = "https://github.com/tersekmatija/lwcc_weights/releases/download/v0.1/{}".format(
        path
    )

    return url


def weights_check(model_name, model_weights):
    # download weights if not available
    home = str(Path.home())

    # create dir if does not exists
    Path(os.path.join(home, ".lwcc/weights")).mkdir(parents=True, exist_ok=True)

    file_name = "{}_{}.pth".format(model_name, model_weights)
    url = build_url(file_name)
    output = os.path.join(home, ".lwcc/weights/", file_name)
    print(output)

    if not os.path.isfile(output):
        print(file_name, " will be downloaded to ", output)
        gdown.download(url, output, quiet=False)

    return output


def load_image(img: img_type, model_name: str, is_gray=False, resize_img=True) -> Image:
    # img type check
    f = None
    if isinstance(img, (str, os.PathLike)):
        if not os.path.isfile(img):
            raise ValueError("Confirm that {} exists".format(img))
        f = open(img, 'rb')
        img = Image.open(f)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # set transform
    if is_gray:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # preprocess image
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # resize image
    if resize_img:
        long = max(img.size[0], img.size[1])
        factor = 1000 / long
        img = img.resize(
            (int(img.size[0] * factor), int(img.size[1] * factor)),
            Image.BILINEAR
        )

    # different preprocessing for SFANet
    if model_name == "SFANet":
        height, width = img.size[1], img.size[0]
        height = round(height / 16) * 16
        width = round(width / 16) * 16
        img = img.resize((width, height), Image.BILINEAR)

    img = trans(img)
    img = img.unsqueeze(0)

    if f is not None:
        f.close()

    return img


def load_image_arr(img_arr, model_name, is_gray=False, resize_img=True):
    """loads an opencv frame as image
    Args:
        img_arr: opencv frame in RGB
        model_name: name of the model
        is_gray (bool, optional): convert to gray scale. Defaults to False.
        resize_img (bool, optional): resize image. Defaults to True.
    """
    # set transform
    if is_gray:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # preprocess image
    img = Image.fromarray(img_arr)

    # resize image
    if resize_img:
        long = max(img.size[0], img.size[1])
        factor = 1000 / long
        img = img.resize((int(img.size[0] * factor), int(img.size[1] * factor)),
                         Image.BILINEAR)

    # different preprocessing for SFANet
    if model_name == "SFANet":
        height, width = img.size[1], img.size[0]
        height = round(height / 16) * 16
        width = round(width / 16) * 16
        img = img.resize((width, height), Image.BILINEAR)

    img = trans(img)
    img = img.unsqueeze(0)

    return img
