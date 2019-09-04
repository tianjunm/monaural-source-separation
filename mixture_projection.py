"""Calculate projection loss"""
import argparse
import torch
import os

import model.dataset as data_util
import model.models as mms
import model.proj as proj
import constants as const
import utils


def get_argument():
    parser = argparse.ArgumentParser(description='generating visualizations')

    parser.add_argument('-g', '--gpu_id', type=int)
    parser.add_argument('-t', '--dataset_type')

    return parser.parse_args()



def get_device(gpu_id):
    """get cuda device object"""
    if torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    return torch.device('cpu')


def compute_loss(model, criterion, batch, device):
    aggregate = batch['aggregate'].to(device)
    ground_truths = batch['ground_truths'].to(device)
    prediction = model(aggregate)
    loss = criterion(prediction, ground_truths)
    return loss.item()


def main():
    args = get_argument()
    device = get_device(args.gpu_id)

    nsrc = int(args.dataset_type.split('-')[1][0])
    print(args.dataset_type)

    root_path = '/home/ubuntu/datasets/processed/datagen'
    # root_path = '/home/tianjunm/Dropbox/projects/multimodal-listener_local/tmp'
    path = os.path.join(root_path, args.dataset_type, 'val.csv')

    tr = data_util.Wav2Spect('Concat')
    ds = data_util.MixtureDataset(
        num_sources=nsrc,
        data_path=path,
        transform=tr)

    bs = 128
    loader0 = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)

    model = proj.PROJ(nsrc)
    criterion = mms.GreedyLoss(device, 'euclidean', nsrc)
    running_loss = 0.0
    iters = 0
    model.eval()
    with torch.no_grad():
        for batch in loader0:
            iters += 1
            loss = compute_loss(model, criterion, batch, device)
            running_loss += loss

    loss = running_loss / iters
    print(f"{args.dataset_type}: {loss}")


main()
