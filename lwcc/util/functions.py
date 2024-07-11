import os
from pathlib import Path
from typing import Callable, Union

import gdown
import numpy as np
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
    Path(os.path.join(home, ".lwcc/weights")).mkdir(
        parents=True, exist_ok=True
    )

    file_name = "{}_{}.pth".format(model_name, model_weights)
    if model_name.startswith("AWCCNet"):
        url = build_url(file_name)
    else:
        url = (
            "https://drive.google.com/uc?id=1Tu9VH0FmWyMTTwe8rqQt3gq_U2mUZGY3"
        )
    output = os.path.join(home, ".lwcc/weights/", file_name)
    print(output)

    if not os.path.isfile(output):
        print(file_name, " will be downloaded to ", output)
        gdown.download(url, output, quiet=False)

    return output


def load_image(
    img: img_type, model_name: str, is_gray=False, resize_img=True
) -> torch.Tensor:
    # img type check
    f = None
    if isinstance(img, (str, os.PathLike)):
        if not os.path.isfile(img):
            raise ValueError("Confirm that {} exists".format(img))
        f = open(img, "rb")
        imag = Image.open(f)
    elif isinstance(img, np.ndarray):
        imag = Image.fromarray(img)

    # set transform
    if is_gray:
        color_norm = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    else:
        color_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    trans: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(color_norm[0], color_norm[1]),
        ]
    )

    # preprocess image
    if imag.mode != "RGB":
        img = imag.convert("RGB")

    # resize image
    if resize_img:
        long = max(imag.size[0], imag.size[1])
        factor = 1000 / long
        img = imag.resize(
            (int(imag.size[0] * factor), int(imag.size[1] * factor)),
            Image.Resampling.BILINEAR,
        )

    # different preprocessing for SFANet
    if model_name == "SFANet":
        height, width = imag.size[1], imag.size[0]
        height = round(height / 16) * 16
        width = round(width / 16) * 16
        imag = imag.resize((width, height), Image.Resampling.BILINEAR)

    img_t = trans(imag)
    img_t = img_t.unsqueeze(0)

    if f is not None:
        f.close()

    return img_t


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
        color_norm = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    else:
        color_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(color_norm[0], color_norm[1]),
        ]
    )

    # preprocess image
    img = Image.fromarray(img_arr)

    # resize image
    if resize_img:
        long = max(img.size[0], img.size[1])
        factor = 1000 / long
        img = img.resize(
            (int(img.size[0] * factor), int(img.size[1] * factor)),
            Image.Resampling.BILINEAR,
        )

    # different preprocessing for SFANet
    if model_name == "SFANet":
        height, width = img.size[1], img.size[0]
        height = round(height / 16) * 16
        width = round(width / 16) * 16
        img = img.resize((width, height), Image.Resampling.BILINEAR)

    img = trans(img)
    img = img.unsqueeze(0)  # type: ignore

    return img
