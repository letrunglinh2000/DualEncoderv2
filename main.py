from networks.dualencoder_v2 import DualEncoderv2
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = DualEncoderv2()
    print("total parameters :",count_parameters(model))
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    print(out.shape)
