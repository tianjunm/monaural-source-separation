"""For visualizing dataset and separation resutls."""
import argparse
import os
import librosa
import librosa.display
import numpy as np
# import scipy
import matplotlib.pyplot as plt

import scipy
import torch
from tqdm import tqdm

import model.transformer as mtr
import model.dataset as data_util
import model.models as mms
import model.proj as mpr
import constants as const
import utils

from mir_eval import separation as sep


FS = 44100
NPERSEG = 256
NOVERLAP = NPERSEG // 4


def get_argument():
    parser = argparse.ArgumentParser(description='generating visualizations')

    parser.add_argument('-g', '--gpu_id', type=int)
    parser.add_argument('-t', '--dataset_type')
    parser.add_argument('-c', '--criterion', default='avg')
    parser.add_argument('-m', '--model_type')

    parser.add_argument(
        '-r',
        '--rank',
        action='store_const',
        const=True,
        default=False)

    return parser.parse_args()


def get_device(gpu_id):
    """get cuda device object"""
    if torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    return torch.device('cpu')


def load_model(model_type, exp_id, config_id, dataset_type, device):
    nsrc = int(dataset_type.split('-')[1][0])
    config_path = os.path.join(
        const.RESULT_PATH_PREFIX,
        f"{dataset_type}_euclidean_{model_type}_{exp_id}",
        config_id,
        'config.tar')

    checkpoint_path = os.path.join(
        const.RESULT_PATH_PREFIX,
        f"{dataset_type}_euclidean_{model_type}_{exp_id}",
        config_id,
        'snapshots/best.tar')

    print(config_path)
    print(checkpoint_path)

    config = torch.load(config_path, map_location=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    m = checkpoint['model_state_dict']
    nm = {}
    for key in m.keys():
        if key.startswith('module'):
            nm[key[7:]] = m[key]
        else:
            nm[key] = m[key]

    if model_type == 'PROJ':
        net = mpr.PROJ(nsrc)

    elif model_type == 'GAB':
        net = mms.LookToListenAudio(
            input_dim=129,
            num_sources=nsrc,
            chan=config['chan']).to(device)
        net.load_state_dict(nm)

    elif model_type == 'VTF':
        net = mtr.make_model(
            input_dim=129 * 2,
            N=config['N'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            h=config['h'],
            num_sources=nsrc,
            dropout=config['dropout']).to(device)
    else:
        try:
            cr_args = {
                'c_out': config['c_out'],
                'd_out': config['d_out'],
                'ks2': config['ks2'],
            }
        except:
            cr_args = {
                'c_out': 512,
                'd_out': 64,
                'ks2': 32
                    }

        net = mtr.make_stt(
            input_dim=129 * 2,
            seq_len=460,
            stt_type="STT1-CR",
            N=config['N'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            h=config['h'],
            num_sources=nsrc,
            dropout=config['dropout'],
            cr_args=cr_args,
            res_size=config['res_size']).to(device)

        net.load_state_dict(nm)

    return net


def create_criterion(device, criterion):
    if criterion == 'avg':
        criterion = mms.ProdGreedy(device, 'euclidean', 2)
    else:
        criterion = mms.GreedyLoss(device, 'euclidean', 2)
    return criterion


def display(spect, sample_rate, hop_length, y_axis='linear', x_axis='time'):
    db_data = librosa.power_to_db(np.abs(spect)**2, ref=np.max)
    librosa.display.specshow(
        db_data,
        sr=sample_rate,
        hop_length=hop_length,
        y_axis=y_axis,
        x_axis=x_axis)


def get_spect_recon(data, nsrc, xid=0):
    _, seq_len, nsrc, input_dim = data.shape

    spects = []
    recons = []
    print(nsrc)
    for i in range(nsrc):
        spect = get_spect(data[xid, :, i].cpu())
        recon = get_wav(spect)

        spects.append(spect)
        recons.append(recon)

    return spects, recons


def get_spect(data, mtype=''):
    if mtype == 'GAB':
        real = data[0].numpy().T
        imag = data[1].numpy().T
    else:
        real = np.split(data.numpy().T, 2)[0]
        imag = np.split(data.numpy().T, 2)[1]

    spect = real + 1j * imag
    return spect


def get_wav(spect):
    _, recon = scipy.signal.istft(spect, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    return recon


# def get_gt_spects(ground_truths_gt, fs, nperseg, noverlap, nsrc, xid=0):
#     # _, seq_len, nsource, input_dim = ground_truths_gt.shape

#     c_gts = []
#     # c_ins = []
#     revs = []
#     for i in range(nsrc):
#         gt = ground_truths_gt[xid, :, i]
#         # gt_in = ground_truths_in.view(-1, seq_len+1, nsource, input_dim)[xid, :, i]
#         gt_spect = get_spect(gt)
#         gt_recon = get_wav(gt_spect)
#         # c_in, rev = get_spect(gt_in, fs, nperseg, noverlap)

#         # print(c_gt.shape)
#         gt_spects.append(gt_spect)
#         # c_ins.append(c_in)
#         gt_recons.append(gt_recon)

#     return c_gts, revs


def compute_loss(model, criterion, batch, device):
    aggregate = batch['aggregate'].to(device)
    in_gts = batch['ground_truths_in'].to(device)
    cmp_gts = batch['ground_truths_gt'].to(device)

    mask_size = in_gts.shape[1]
    subseq_mask = mtr.subsequent_mask(mask_size).to(device)

    prediction = model(aggregate, in_gts, None, subseq_mask)
    loss = criterion(prediction, cmp_gts)

    return loss.item()


def run_inference(model, model_type, nsrc, info, device):
    aggregate = info['aggregate'].unsqueeze(0)
    if model_type in ['STT1-CR', 'VTF']:
        # ground_truths_in = info['ground_truths_in'].unsqueeze(0)
        gts = info['ground_truths_gt'].unsqueeze(0)
        _, seq_len, input_dim = aggregate.shape

        # subseq_mask = mtr.subsequent_mask(ground_truths_in.shape[1])

        seps = mtr.greedy_decoder(model, aggregate.to(device), seq_len, nsrc, input_dim, device)
        outputs = torch.stack(torch.split(seps, 258, dim=-1), dim=2)

    else:
        gts = info['ground_truths'].unsqueeze(0)
        outputs = model(aggregate.to(device))

    return aggregate, gts, outputs


# FIXME
def get_stats(model, dataloader, device, nsrc, mtype='stt'):
    # baseline = mpr.PROJ(nsrc)
    # baseline.eval()
    model.eval()
    final_sdr = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(i)

            if mtype != 'stt':
                mixture = batch['aggregate']
                refs = batch['ground_truths']
                outputs = model(mixture.to(device))

            else:
                mixture = batch['aggregate']
                refs = batch['ground_truths_gt']
                _, sl, dim = mixture.size()
                seps = mtr.greedy_decoder(model, mixture.to(device), sl, nsrc, dim, device)
                outputs = torch.stack(torch.split(seps, dim, dim=-1), dim=2)

            ref_spects, ref_recons = get_spect_recon(refs, nsrc)
            est_spects, est_recons = get_spect_recon(outputs, nsrc)

            corr = mms.get_loss(refs.cpu(), outputs.cpu(), device, 'correlation')
            # sep.validate(est_recons, ref_recons)
            # sdr, sir, sar, _ = sep.bss_eval_sources(ref_recons, est_recons)
            final_sdr += corr.mean()
            print(final_sdr / (i + 1))
            # return ref_recons, est_recons
        # break
        # return final_sdr / 2000


def save_info(spects, audios, categories, idx, dataset_type, mtype, criterion, rec=False):
    for j, c in enumerate(spects):
        # try:
        # normalize audio
        audio = audios[j]
        audio = audio / np.max(np.abs(audio), axis=0)
        # except:
            # audio = audios[j][0]
        # audio = audios[j]
        # visualizations/results/t3-2s-25c/0/sss
        if rec and j > 0:
            outname = f"{j}_rec"
        else:
            outname = str(j)

        out_path = os.path.join(
            const.VRESULT_PATH,
            dataset_type,
            mtype,
            criterion,
            idx,
            f"{outname}.png")

        audio_path = os.path.join(
            const.VRESULT_PATH,
            dataset_type,
            mtype,
            criterion,
            idx,
            f"{outname}.wav")

        utils.make_dir(out_path)
        utils.make_dir(audio_path)

        # save audio
        scipy.io.wavfile.write(audio_path, FS, audio)
        # librosa.output.write_wav(audio_path, audio, FS)

        fig = plt.figure(figsize=(8, 2))
        if j == 0:
            plt.title('mixture')
        else:
            if not rec:
                category = categories[j - 1]
                plt.title(category)
            else:
                plt.title(' ')

        display(c, FS, hop_length=NPERSEG * 3 // 4)

        plt.colorbar(format='%+2.0f dB')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

        plt.close(fig)


def main():
    args = get_argument()

    device = get_device(args.gpu_id)
    # load dataset
    root_path = '/home/ubuntu/datasets/processed/datagen'
    # root_path = '/home/tianjunm/Dropbox/projects/multimodal-listener_local/tmp'
    path = os.path.join(root_path, args.dataset_type, 'train.csv')

    print(path)

    nsrc = int(args.dataset_type.split('-')[1][0])

    trs = {
        'STT1-CR': data_util.Wav2Spect('Concat', enc_dec=True),
        'VTF': data_util.Wav2Spect('Concat', enc_dec=True),
        'GAB': data_util.Wav2Spect()
    }

    ds0 = data_util.MixtureDataset(
        num_sources=nsrc,
        data_path=path,
        transform=trs[args.model_type])

    bs = 1
    loader0 = torch.utils.data.DataLoader(ds0, batch_size=bs, shuffle=True)
    # loader1 = torch.utils.data.DataLoader(ds1, batch_size=bs, shuffle=False)

    print(f"set up {args.model_type}")
    model = load_model(args.model_type, '190820', '0', args.dataset_type, device)
    criterion = create_criterion(device, args.criterion)

    print(f"finished set up {args.model_type}")
    # rank top 25 STT examples

    if args.rank:
        losses = []

        for i, batch in enumerate(loader0):
            if i == 2000:
                break
            loss = compute_loss(model, criterion, batch, device)
            losses.append((loss, i))

        torch.save({'losses': losses}, f'losses_{args.criterion}.tar')

    else:
        losses = torch.load(f'losses_{args.criterion}.tar')['losses']

    # tops = sorted(losses)[:100]

    tops = [503, 537, 562, 625, 901, 943, 1113, 1690]
    # tops = [3366, 9, 1905, 2758, 5901, 7478, 9413, 9453, 12574]
    # tops = [5901]

    print(tops)
    model.eval()
    with torch.no_grad():
        # for i, (_, idx) in enumerate(tqdm(tops)):
        for i, idx in enumerate(tqdm(tops)):
            info = ds0[idx]
            categories = info['category_names']

            aggregate, gts, outputs = run_inference(model, args.model_type, nsrc, info, device)

            # mixture
            mspect = get_spect(aggregate[0], args.model_type)
            maudio = get_wav(mspect)

            # gt
            gtspects, gtaudio = get_spect_recon(gts, nsrc)
            # outputs
            recspects, recaudio = get_spect_recon(outputs, nsrc)

            # print(len(gtspects))
            # print(gtaudio.shape)
            # save original spects
            save_info(
                [mspect] + gtspects,
                [maudio] + gtaudio,
                categories,
                str(idx),
                args.dataset_type,
                args.model_type,
                args.criterion)

            # save recs
            save_info(
                [mspect] + recspects,
                [maudio] + recaudio,
                categories,
                str(idx),
                args.dataset_type,
                args.model_type,
                args.criterion, rec=True)

        # break

    # losses = []
    # found = 0
    # for _, batch in enumerate(loader0):
    #     mixture = batch['aggregate']
    #     ground_truths_in = batch['ground_truths_in']
    #     ground_truths_gt = batch['ground_truths_gt']
    #     categories = batch['category_names']

    #     if found >= 6:
    #         break

    #     # create spect and save
    #     for i in tqdm(range(bs)):
    #         # seen = set()
    #         # for s in range(nsrc):
    #         #     seen.add(categories[s][i])

    #         # if len(seen) < nsrc:
    #             found += 1

    #             c_agg = get_spect(
    #                 mixture[i],
    #                 fs=FS,
    #                 nperseg=NPERSEG,
    #                 noverlap=NOVERLAP)
    #             c_gts = get_gt_spect(
    #                 ground_truths_gt,
    #                 fs=FS,
    #                 nperseg=NPERSEG,
    #                 noverlap=NOVERLAP,
    #                 nsrc=nsrc,
    #                 xid=i)
    #             for j, c in enumerate([c_agg] + c_gts):
    #                 out_path = os.path.join(
    #                     const.VDATA_PATH,
    #                     args.dataset_type,
    #                     f"{found}_{j}.png")
    #                 utils.make_dir(out_path)

    #                 plt.figure(figsize=(8, 2))
    #                 if j == 0:
    #                     plt.title('mixture')
    #                 else:
    #                     category = categories[j - 1][i]
    #                     plt.title(category)


    #                 display(c, FS, hop_length=NPERSEG * 3 //4)

    #                 plt.colorbar(format='%+2.0f dB')
    #                 plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

    #     break

        # compute loss and store top diff ids

if __name__ == "__main__":
    main()
