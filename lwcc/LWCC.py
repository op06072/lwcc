from lwcc.models import CSRNet, SFANet, Bay, DMCount
from lwcc.models.AWCC_Net import CC as AWCCNet
from lwcc.util.functions import (
    load_image,
    load_image_arr,
    img_type,
    tensor_convert,
)

import os
import torch
import numpy as np
from PIL import Image
from platform import system

loaded_models: dict[str, torch.nn.Module] = {}


def load_model(
    model_name="CSRNet",
    model_weights="SHA",
    path: str | os.PathLike | None = None,
) -> torch.nn.Module:
    """
    Builds a model for Crowd Counting and initializes it as a singleton.
    :param model_name: One of the available models: CSRNet.
    :param model_weights: Name of the dataset the model was pretrained on. Possible values vary on the model.
    :return: Built Crowd Counting model initialized with pretrained weights.
    """

    if path:
        model = torch.load(path)
        return model
    else:
        available_models = {
            "CSRNet": CSRNet,
            "SFANet": SFANet,
            "Bay": Bay,
            "DM-Count": DMCount,
            "AWCCNet": AWCCNet,
        }

        global loaded_models

        if "loaded_models" not in globals():
            loaded_models = {}

        model_full_name = "{}_{}".format(model_name, model_weights)
        if model_full_name not in loaded_models.keys():
            model = available_models.get(model_name)
            if model:
                model = model.make_model(model_weights)
                loaded_models[model_full_name] = model
                print(
                    "Built model {} with weights {}".format(
                        model_name, model_weights
                    )
                )
            else:
                raise ValueError(
                    "Invalid model_name. Model {} is not available.".format(
                        model_name
                    )
                )

        return loaded_models[model_full_name]


def get_count(
    images: img_type | list[img_type],
    model_name="CSRNet",
    model_weights="SHA",
    model=None,
    is_gray=False,
    return_density=False,
    resize_img=True,
    device="cpu",
    compile_model=False,
):
    """
    Return the count on image/s. You can use already loaded model or choose the name and pre-trained weights.
    :param images: Either a String (path to the image) or a numpy array (OpenCV frame) or a PIL Image or a list of them.
    :param model_name: If not using preloaded model, choose the model name. Default: "CSRNet".
    :param model_weights: If not using preloaded model, choose the model weights.  Default: "SHA".
    :param model: Possible preloaded model. Default: None.
    :param is_gray: Are the input images grayscale? Default: False.
    :param return_density: Return the predicted density maps for input? Default: False.
    :param resize_img: Should images with high resolution be down-scaled? This is especially good for high resolution
            images with relatively few people. For very dense crowds, False is recommended. Default: True
    :param device: Which device to use for Pytorch. Default: "cpu".
    :return: Depends on whether the input is a String or list and on the return_density flag.
        If input is a String, the output is a float with the predicted count.
        If input is a list, the output is a dictionary with image names as keys, and predicted counts (float) as values.
        If return_density is True, function returns a tuple (predicted_count, density_map).
        If return_density is True and input is a list, function returns a tuple (count_dictionary, density_dictionary).
    """

    # if one path to array
    img_arr: list[img_type] = []
    if isinstance(images, (str, os.PathLike, np.ndarray, Image.Image)):
        img_arr = [images]
    else:
        img_arr = images

    # load model
    if model is None:
        model = load_model(model_name, model_weights)

    # load images
    imgs, names = [], []

    for idx, img in enumerate(img_arr):
        img_t = load_image(img, model.get_name(), is_gray, resize_img)
        imgs.append(img_t)
        name = idx
        names.append(name)

    if device != "cpu":
        # print(f"transfering tensor to device: {device}")
        imgs = [img.to(device=device) for img in imgs]
        model = model.to(device=device)

    if (
        not torch.backends.mps.is_available()
        and system() != "Windows"
        and compile_model
    ):
        model.compile()

    imgs_t = torch.cat(imgs)

    # with torch.set_grad_enabled(False):
    with torch.no_grad():
        outputs = model(imgs_t)

    counts = tensor_convert(torch.sum(outputs, (1, 2, 3)))
    counts = dict(zip(names, counts))

    densities = dict(zip(names, tensor_convert(outputs[:, 0, :, :])))

    if len(counts) == 1:
        if return_density:
            return counts, densities
        else:
            return counts

    if return_density:
        return counts, densities

    return counts


def get_count_arr(
    img_arr,
    model_name="CSRNet",
    model_weights="SHA",
    model=None,
    is_gray=False,
    resize_img=True,
    device=None,
):
    # load model
    if model is None:
        model = load_model(model_name, model_weights)

    img = load_image_arr(img_arr, model.get_name(), is_gray, resize_img)

    if device is not None:
        print(f"transfering tensor to device: {device}")
        img = img.to(device=device)

    with torch.set_grad_enabled(False):
        output = model(img)

    count = torch.sum(output)
    density = output.cpu().numpy()
    density = density.reshape((density.shape[2], density.shape[3]))

    return float(count.cpu().numpy()), density
