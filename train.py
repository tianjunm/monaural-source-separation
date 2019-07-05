"""train.py

"""
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from model import models, transformer
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
    parser.add_argument('--learn_mask', type=int, default=1)
    parser.add_argument('--dataset', type=str, required=True)
    return parser.parse_args()


def get_device(gpu_id):
    '''get device info'''
    device = 'cpu'
    gpu_name = 'cuda:{}'.format(gpu_id)
    if torch.cuda.is_available():
        device = torch.device(gpu_name)

    return device


def run_epoch(data, model, model_name, n_sources, input_dim, device, learn_mask, 
        criterion):
    aggregate = data['aggregate'].to(device)
    if model_name == 'transformer':
        ground_truths_in = data['ground_truths_in'].to(device)
        ground_truths = data['ground_truths_gt'].to(device)
        mask_size = ground_truths_in.shape[1]
        subseq_mask = transformer.subsequent_mask(mask_size).to(device)
        out = model(aggregate, ground_truths_in, None, subseq_mask)
        prediction = model.generator(aggregate, out, learn_mask=learn_mask)
        loss = criterion(prediction, ground_truths)
    else:
        ground_truths = data['ground_truths'].to(device)
        prediction, _ = model(aggregate)
        loss = criterion(prediction, ground_truths)
    
    return loss

def main():
    args = get_argument()
    device = get_device(args.gpu_id)

    print('Preparing data...', end='')
    train_dir = args.train_dir
    test_dir = args.test_dir
    
    spect_dim = tuple(args.spect_dim) 
    # input_dim = args.num_sources * spect_dim[0] 
    input_dim = 2 * spect_dim[0] 
   
    "data loading function"
    transform = dt.Concat(spect_dim)

    "network creation"
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
        transform = dt.ToTensor(spect_dim)
        model = models.LookListen_Base(
                seq_len=spect_dim[1],
                input_dim=spect_dim[0],
                num_sources=args.num_sources).to(device)

    elif args.model == 'transformer':
        transform = dt.Concat(size=spect_dim, encdec=True)
        # freq_range, seq_len = spect_dim
        model = transformer.make_model(input_dim,
                num_sources=args.num_sources).to(device)
    else:
        model = models.Baseline(
                input_dim, 
                seq_len=spect_dim[1],
                num_sources=args.num_sources).to(device)

    "create dataset"
    dataset = dt.SignalDataset(root_dir=train_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=True)

    # testset = dt.SignalDataset(root_dir=test_dir, transform=transform)
    # testloader = torch.utils.data.DataLoader(
    #         testset, 
    #         batch_size=args.batch_size)

    print('done!')
    # customized loss function
    if args.criterion == 'minloss':
        criterion = models.MinLoss(device, args.metric)
    else: 
        # TODO: currently unavailable due to dimension mismatch
        criterion = models.MSELoss(device, args.metric)

    # FIXME: trying random configs for optimizers
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    # tensorboard writer
    writer = SummaryWriter(logdir=args.output_dir)

    model_path_prefix = '{}/{}'.format(args.model_dir, args.model)
   
    # for early stopping
    # curr_loss = 10000
    # prev_model = None
    
    print('Start training...')
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        test_loss = 0.0 
        for i, info in enumerate(dataloader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            
            loss = run_epoch(info, model, args.model, args.num_sources,
                    input_dim, device, args.learn_mask, criterion) 
            train_loss += loss
            loss.backward()
            optimizer.step()

        # with torch.no_grad():
        #     for test_data in testloader:
        #         test_loss += run_epoch(test_data, model, args.model, 
        #                 args.num_sources, input_dim, device, args.learn_mask,
        #                 criterion)

        # log loss in graph
        legend_train = 'train_{}_{}_{}({})'.format(args.model, args.dataset,
                args.criterion, args.job_id)
        # legend_test = 'validation loss'
        avg_loss = train_loss / len(dataset)
        # avg_test_loss = test_loss / len(testset)

        if epoch == 0 or epoch % 50 == 0:
            model_path = model_path_prefix + '_checkpoint_{}_{}_{}_{}({}).pth'.format(
                    args.dataset, args.criterion, args.metric, epoch, args.job_id)
            torch.save(model.state_dict(), model_path)
            # curr_loss = avg_test_loss

        # print('epoch %d, train loss: %.3f, val loss: %.3f' % \
        #         (epoch + 1, avg_loss, avg_test_loss))

        print('epoch %d, train loss: %.3f' % \
                (epoch + 1, avg_loss))

        # writer.add_scalars('data/{}_loss_{}_{}({})'.format(args.model,
        #     args.metric, args.dataset, args.job_id),
        writer.add_scalars('data/loss_{}'.format(args.metric),
            {
               legend_train: avg_loss,
               # legend_test: avg_test_loss,
            }, epoch)

    print("Finished training!")
    writer.close()


if __name__ == '__main__':
    main()
