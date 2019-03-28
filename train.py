"""train.py

"""
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from model import models
import model.dataset as dt

BATCH_SIZE = 32 
LEARNING_RATE = 1e-3
OUTPUT_DIR = './logs'
NUM_EPOCHS = 100


def get_argument():
    """create parser for arguments"""
    parser = argparse.ArgumentParser(description='Separator network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    # parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    return parser.parse_args()


def get_device(gpu_id):
    """assigns device string"""

    device = 'cpu'
    gpu_name = 'cuda:{}'.format(gpu_id)
    if torch.cuda.is_available():
        device = torch.device(gpu_name)

    return device


def main():
    """trains the selected network"""
    args = get_argument()

    # cuda
    device = get_device(args.gpu_id)
    # create network
    model = models.Baseline(1025, 
            seq_len=173,
            num_sources=2).to(device)

    # customized loss function
    criterion = models.MinLoss(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    
    print('Preparing data...', end='')
    dataset = dt.SignalDataset(root_dir='data/a1_spectrograms/')
    dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=True)
    print('done!')

    # tensorboard writer
    writer = SummaryWriter(log_dir=args.output_dir)

    print('Start training...')
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        epoch_loss = 0.0
        print('epoch {}'.format(epoch))
        for i, info in enumerate(dataloader):
            print('batch {}'.format(i))
            aggregate = info['aggregate'].to(device)
            ground_truths = [gt.to(device) for gt in info['ground_truths']]
            optimizer.zero_grad()
            
            prediction = model(aggregate)
            loss = criterion(prediction, ground_truths)
            print(loss)
            loss.backward()
            optimizer.step()

            # log statistics
            running_loss += loss
            epoch_loss += loss

            # log every 200 mini-batches
            if (i > 0 and (i + 1) % 200 == 0):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                # epoch_loss += running_loss
                running_loss = 0

        # log loss in graph
        legend = 'train loss'
        avg_loss = epoch_loss / len(dataset)
        writer.add_scalars('data/loss',
                           {
                               legend: avg_loss,
                           },
                           epoch)
        epoch_loss = 0
    
    print("Finished training!")
    writer.close()


if __name__ == '__main__':
    main()
