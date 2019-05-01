"""train.py

"""
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from model import models
import model.dataset as dt

# default parameters
NUM_SOURCES = 2
BATCH_SIZE = 32 
LEARNING_RATE = 1e-3
CRITERION = 'minloss'
TRAIN_DIR = '/home/ubuntu/dataset/dataset_c2/train_c2'
TEST_DIR = '/home/ubuntu/dataset/dataset_c2/test_c2'
OUTPUT_DIR = './logs'
NUM_EPOCHS = 3000
MODEL_TYPE = 'base'
METRIC = 'euclidean'


def get_argument():
    """create parser for arguments"""
    parser = argparse.ArgumentParser(description='Separator network')
    parser.add_argument('--spect_dim', type=int, nargs='+')
    parser.add_argument('--num_sources', type=int, default=NUM_SOURCES) 
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--criterion', type=str, default=CRITERION)
    parser.add_argument('--model', type=str, default=MODEL_TYPE)
    parser.add_argument('--metric', type=str, default=METRIC)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR)
    parser.add_argument('--test_dir', type=str, default=TEST_DIR)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    return parser.parse_args()


def get_device(gpu_id):
    '''get device info'''
    device = 'cpu'
    gpu_name = 'cuda:{}'.format(gpu_id)
    if torch.cuda.is_available():
        device = torch.device(gpu_name)

    return device


def main():
    args = get_argument()

    # cuda
    device = get_device(args.gpu_id)

    print('Preparing data...', end='')
    train_dir = args.train_dir
    test_dir = args.test_dir
    
    spect_dim = tuple(args.spect_dim) 
    input_dim = args.num_sources * spect_dim[0] 
    
    # FIXME: bad layout
    if args.model == 'google':
        transform = dt.ToTensor(spect_dim)
    else:
        transform = dt.Concat(spect_dim)

    dataset = dt.SignalDataset(root_dir=train_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=True)

    testset = dt.SignalDataset(root_dir=test_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=args.batch_size)

    print('done!')

    # create network
    if args.model == 'A1':
        model = models.A1(
                input_dim,
                seq_len=spect_dim[1],
                num_sources=args.num_sources).to(device)

    elif args.model == 'A2':
        model = models.B1(
                input_dim,
                num_layers=2,
                seq_len=spect_dim[1],
                num_sources=args.num_sources).to(device)

    elif args.model == 'B1':
        model = models.B1(
                input_dim,
                seq_len=spect_dim[1],
                num_sources=args.num_sources).to(device)
    
    elif args.model == 'google':
        model = models.LookListen_Base(
                seq_len=spect_dim[1],
                input_dim=spect_dim[0]).to(device)
    else:
        model = models.Baseline(
                input_dim, 
                seq_len=spect_dim[1],
                num_sources=args.num_sources).to(device)

    # customized loss function
    if args.criterion == 'minloss':
        criterion = models.MinLoss(device, args.metric)
    else: 
        # TODO: currently unavailable due to dimension mismatch
        criterion = models.MSELoss(device, args.metric)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # tensorboard writer
    writer = SummaryWriter(log_dir=args.output_dir)

    model_path_prefix = '{}/{}'.format(args.model_dir, args.model)
   
    # for early stopping
    # curr_loss = 10000
    # prev_model = None

    print('Start training...')
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        test_loss = 0.0 
        for i, info in enumerate(dataloader):
            aggregate = info['aggregate'].to(device)
            ground_truths = info['ground_truths'].to(device)
            optimizer.zero_grad()

            prediction, _ = model(aggregate)
            loss = criterion(prediction, ground_truths)
            train_loss += loss
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for test_data in testloader:
                aggregate = test_data['aggregate'].to(device)
                ground_truths = test_data['ground_truths'].to(device)
                prediction, _ = model(aggregate)
                loss = criterion(prediction, ground_truths)
                test_loss += loss

        # log loss in graph
        legend_train = 'train loss'
        legend_test = 'validation loss'
        avg_loss = train_loss / len(dataset)
        avg_test_loss = test_loss / len(testset)

        if epoch % 49 == 0:
            model_path = model_path_prefix + '_checkpoint_{}_{}_{}.pth'.format(args.metric, args.job_id, epoch)
            torch.save(model.state_dict(), model_path)
            # curr_loss = avg_test_loss

        print('epoch %d, train loss: %.3f, val loss: %.3f' % (epoch + 1, avg_loss, avg_test_loss))

        writer.add_scalars('data/{}_loss_{}_{}'.format(args.model, args.metric, args.job_id),
                           {
                               legend_train: avg_loss,
                               legend_test: avg_test_loss,
                           },
                           epoch)

    print("Finished training!")
    writer.close()


if __name__ == '__main__':
    main()
