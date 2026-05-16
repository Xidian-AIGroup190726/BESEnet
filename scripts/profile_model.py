import time

import torch
from thop import profile

from models.dpcd_network import ProfessionalChangeDetectionNetwork


def profile_network_runtime():
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = ProfessionalChangeDetectionNetwork().to(device)
    reference_input = torch.randn(1, 3, 256, 256).to(device)

    flops, parameters = profile(network, inputs=(reference_input, reference_input))
    print("flops: ", flops, "params: ", parameters)

    with torch.no_grad():
        start_time = time.time()
        network(reference_input, reference_input)
        inference_time = time.time() - start_time

    print(f"Inference time: {inference_time:.4f} seconds")


if __name__ == "__main__":
    profile_network_runtime()
