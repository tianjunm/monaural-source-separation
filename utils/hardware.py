import torch



def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def get_all_devices():
    return torch.cuda.device_count()