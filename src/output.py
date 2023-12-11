import random
import time

import progressbar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import models
import utils
from data_loader import *
from params import Params


class Pipeline:
    def __init__(self, args, media, idx2word, word2idx):
        self.args = args
        self.epoch = 0
        self.mse = nn.MSELoss().to(self.args.device)
        self.l1 = nn.L1Loss().to(self.args.device)
        self.bce = nn.BCELoss().to(self.args.device)
        self.ce = nn.CrossEntropyLoss().to(self.args.device)
        self.real_label = 1
        self.fake_label = 0
        self.idx2word = idx2word # function
        self.word2idx = word2idx # function
        self.norm_inv = utils.NormalizeInverse((0.5,), (0.5,))
        self.media = media
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        if self.media == 'image':
            self.enc = models.__dict__['attn_trace_encoder_%d' % self.args.trace_w](dim=self.args.nz, nc=self.args.trace_c)
            self.enc = self.enc.to(self.args.device)
            if self.args.use_refiner:
                self.dec = models.__dict__['ResDecoder128'](dim=self.args.nz, nc=3)
                self.dec = self.dec.to(self.args.device)
                self.refiner = models.__dict__['RefinerG_BN'](nc=3, ngf=self.args.nz)
                self.refiner = self.refiner.to(self.args.device)
            else:
                # self.dec = models.__dict__['image_decoder_128'](dim=self.args.nz, nc=3)
                self.dec = models.__dict__['ResDecoder128'](dim=self.args.nz, nc=3)
                self.dec = self.dec.to(self.args.device)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path, map_location=self.args.device)
        self.enc.load_state_dict(ckpt['enc'])
        self.dec.load_state_dict(ckpt['dec'])

    def load_refiner(self, path):
        ckpt = torch.load(path, map_location=self.args.device)
        self.refiner.load_state_dict(ckpt['G'])

    def save_image(self, output, name_list, path):
        assert len(output) == len(name_list)
        for i in range(len(output)):
            utils.save_image(output[i].unsqueeze(0).data,
                             path + name_list[i] + '.jpg',
                             normalize=True, nrow=1, padding=0)

    def output(self, data_loader, recons_dir, target_dir):
        with torch.no_grad():
            if self.media == 'text':
                record = utils.Record()
                record_acc = utils.Record()
                start_time = time.time()
                progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
                for i, data in enumerate(data_loader):
                    progress.update(i + 1)
                    (trace, indexes, sent_length, *_) = data
                    trace = trace.to(self.args.device)
                    indexes = indexes.to(self.args.device)
                    sent_length = sent_length.to(self.args.device)
                    
                    encoded = self.enc(trace)

                    scores, caps_sorted, decode_lengths, alphas, sort_ind = self.dec(encoded, indexes, sent_length)
                    targets = caps_sorted[:, 1:]
                    
                    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                    scores = scores.data
                    targets = targets.data
                    
                    loss = self.ce(scores, targets)
                    record.add(loss.item())
                    topk = 1
                    acc = utils.accuracy(scores.detach(), targets.detach(), topk)
                    record_acc.add(float(acc))
                progress.finish()
                utils.clear_progressbar()
                print('------------------%s------------------' % (self.media.capitalize()))
                print('Cost Time: %f' % (time.time() - start_time))
                print('Loss: %f' % record.mean())
                print('Top %d Acc: %f' % (topk, record_acc.mean()))
                print('----------------------------------------')
            else:
                start_time = time.time()
                progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
                for i, data in enumerate(data_loader):
                    progress.update(i + 1)
                    (trace, image, name_list, *_) = data
                    trace = trace.to(self.args.device)
                    image = image.to(self.args.device)
                    encoded = self.enc(trace)
                    decoded = self.dec(encoded)
                    if self.media == 'image':
                        if self.args.use_refiner:
                            refined = self.refiner(decoded)
                            self.save_image(refined.detach(), name_list, recons_dir)
                        else:
                            self.save_image(decoded.detach(), name_list, recons_dir)
                        self.save_image(image, name_list, target_dir)
                    else:
                        self.save_audio(decoded.detach(), name_list, recons_dir)
                        self.save_audio(image, name_list, target_dir)
                progress.finish()
                utils.clear_progressbar()
                print('------------------%s------------------' % (self.media.capitalize()))
                print('Cost Time: %f' % (time.time() - start_time))
                print('Reconstructed %s Saved in %s' % (self.media, recons_dir))
                print('Target %s Saved in %s' % (self.media, target_dir))

if __name__ == '__main__':
    args = Params().parse()

    ROOT = '..'

    dataset2media = {
        'CelebA': 'image',
        'CIFAR100': 'image'
    }

    media2trace = {
        'image': (6, 256),
        'audio': (8, 512),
        'text': (6, 128)
    }

    media2nz = {
        'image': 128,
        'audio': 256,
        'text': 512
    }

    media = dataset2media[args.dataset]
    (args.trace_c, args.trace_w) = media2trace[media]
    args.nz = media2nz[media]

    args.use_refiner = 1
    args.exp_name = 'recons_%s' % args.dataset
    print(args.exp_name)

    manual_seed = random.randint(1, 10000)
    print('Manual Seed: %d' % manual_seed)

    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    utils.make_path(args.output_root)
    utils.make_path(os.path.join(args.output_root, args.exp_name))

    args.ckpt_root = os.path.join(args.output_root, args.exp_name, 'ckpt')
    args.recons_root = os.path.join(args.output_root, args.exp_name, 'recons')
    args.target_root = os.path.join(args.output_root, args.exp_name, 'target')

    utils.make_path(args.ckpt_root)
    utils.make_path(args.recons_root)
    utils.make_path(args.target_root)

    test_dataset = CelebaDataset(
                img_dir=args.data_path[args.dataset]['media'], 
                npz_dir=args.data_path[args.dataset][args.side],
                ID_path=args.data_path[args.dataset]['ID_path'],
                split=args.data_path[args.dataset]['split'][1],
                trace_c=args.trace_c,
                trace_w=args.trace_w,
                image_size=args.image_size,
                side=args.side
            )
    if args.use_refiner:
        ckpt_path = os.path.join(ROOT, '/models/pin/CelebA_cacheline_pre/final.pth')
        refiner_path = os.path.join(ROOT, '/models/pin/CelebA_refiner/refiner-final.pth')
    else:
        ckpt_path = os.path.join(ROOT, '/models/pin/CelebA_cacheline/final.pth')

    loader = DataLoader(args)

    test_loader = loader.get_loader(test_dataset, shuffle=False)

    engine = Pipeline(args, media, idx2word=None, word2idx=None)
    
    # Use our trained models
    engine.load_model(ckpt_path)
    if args.use_refiner:
        engine.load_refiner(refiner_path)

    # # Use your models
    # engine.load_model((args.ckpt_root + 'final.pth'))
    # if media == 'image' and args.use_refiner:
    #     engine.load_refiner(args.data_path[args.dataset]['refiner_path'])
    
    engine.output(test_loader,
                recons_dir=args.recons_root,
                target_dir=args.target_root)
