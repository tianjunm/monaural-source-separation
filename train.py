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
CRITERION = 'minloss'
OUTPUT_DIR = './logs'
NUM_EPOCHS = 10000
MODEL_TYPE = 'base'


def get_argument():
    """create parser for arguments"""
    parser = argparse.ArgumentParser(description='Separator network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--criterion', type=str, default=CRITERION)
    parser.add_argument('--model', type=str, default=MODEL_TYPE)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--job_id', type=int, default=0)
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
<<<<<<< Updated upstream
    model = models.Baseline(1025, 
            seq_len=173,
            num_sources=2).to(device)
=======
    if args.model == 'base':
        model = models.Baseline(258, 
                seq_len=691,
                num_sources=2).to(device)
    
    else:
        model = models.A1(258, 
                # seq_len=691,
                seq_len=691,
                num_sources=2).to(device)
>>>>>>> Stashed changes

    # customized loss function
    if args.criterion == CRITERION:
        criterion = models.MinLoss(device)
    else: 
        # TODO: currently unavailable due to dimension mismatch
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print('Preparing data...', end='')
    # root_dir = '/home/tianjunm/Documents/Projects/dataset/b2_spectrograms/'
    root_dir = '/home/tianjunm/Documents/Projects/dataset/a1_spectrograms/'
    # test_dir = '/home/tianjunm/Documents/Projects/dataset/b2_spectrograms_test/'
    # root_dir = '/Users/tianjunma/Projects/dataset/a1_spectrograms/'
    dataset = dt.SignalDataset(root_dir=root_dir)
    dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=True)
    # testset = dt.SignalDataset(root_dir=root_dir)
    # testloader = torch.utils.data.DataLoader(
    #         dataset, 
    #         batch_size=args.batch_size,
    #         shuffle=True)
    print('done!')

    # tensorboard writer
    writer = SummaryWriter(log_dir=args.output_dir)

    model_path_prefix = 'pretrained/{}'.format(args.model)

    print('Start training...')
    for epoch in range(NUM_EPOCHS):
        # running_loss = 0.0
        epoch_loss = 0.0
        # print('epoch {}'.format(epoch))
        for i, info in enumerate(dataloader):
            # print('batch {}'.format(i))
            aggregate = info['aggregate'].to(device)
            ground_truths = [gt.to(device) for gt in info['ground_truths']]
            optimizer.zero_grad()
            
            prediction = model(aggregate)
            loss = criterion(prediction, ground_truths)
            # print(loss)
            loss.backward()
            optimizer.step()

            # log statistics
            # running_loss += loss
            epoch_loss += loss

            # log every 20 mini-batches
            # if (i > 0 and (i + 1) % 20 == 0):
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 20))
            #     # epoch_loss += running_loss
            #     running_loss = 0
        
        # with torch.no_grad():
        #     for data in testloader:
        #         aggregate = data['aggregate'].to(device)
        #         ground_truths = [gt.to(device) for gt in info['ground_truths']]
        #         prediction = model(aggregate)
        #         test_loss = criterion(prediction, ground_truths)
                
        if epoch % 49 == 0:
            model_path = model_path_prefix + 'model_checkpoint_{}.pth'.format(epoch + 1)
            torch.save(model.state_dict(), model_path)

        # log loss in graph
        legend_train = 'train loss'
        # legend_test = 'test loss'
        avg_loss = epoch_loss / len(dataset)
        # avg_test_loss = test_loss / len(testset)

        # print('epoch %d, train loss: %.3f, test loss: %.3f' % (epoch + 1,
        #     avg_loss, avg_test_loss))

        print('epoch %d, train loss: %.3f' % (epoch + 1, avg_loss))

        writer.add_scalars('data/{}_loss_{}'.format(args.model, args.job_id),
                           {
                               legend_train: avg_loss,
                               # legend_test: avg_test_loss,
                           },
                           epoch)
        epoch_loss = 0
    
    print("Finished training!")
    writer.close()


if __name__ == '__main__':
    main()
