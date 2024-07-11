from lwcc.util.functions import weights_check

import torch
import torch.nn as nn


def make_model(model_weights) -> nn.Module:
    weights_path = weights_check("AWCCNet", "multi")

    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))

    model = CrowdCounter()
    model.load_state_dict(state_dict["state_dict"])

    return model


class CrowdCounter(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        # model
        from lwcc.models.AWCC_Net.AWCC import vgg19_trans

        self.net = vgg19_trans(use_pe=True)

    def get_name(self):
        return "AWCCNet"

    @torch.no_grad()
    def forward(self, x):
        input_list = [x]
        dmap_list = []
        for inp in input_list:
            pred_map, _ = self.net(inp)
            dmap_list.append(pred_map.detach())
        return torch.relu(dmap_list[0])
