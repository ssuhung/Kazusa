import random
import time

import progressbar
import torch
import torch.nn as nn

import models
import utils
from data_loader import *
from params import Params
from utils import Printer


class ImageEngine:
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.real_label = 1
        self.fake_label = 0
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        # self.enc = models.__dict__['attn_trace_encoder_%d' % self.args.trace_w](dim=self.args.nz, nc=self.args.trace_c)
        # self.enc = models.__dict__['trace_encoder_%d' % self.args.trace_w](dim=self.args.nz, nc=self.args.trace_c)
        self.enc = models.TraceEncoder_1DCNN_encode(input_len=300000, dim=self.args.nz)
        self.enc = self.enc.to(self.args.device)

        self.dec = models.__dict__['ResDecoder%d' % self.args.image_size](dim=self.args.nz, nc=self.args.nc)
        # self.dec = models.__dict__['image_decoder_%d' % self.args.image_size](dim=self.args.nz, nc=self.args.nc)
        self.dec = self.dec.to(self.args.device)    

        self.optim = torch.optim.Adam(
                        list(self.enc.parameters()) + \
                        list(self.dec.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

        self.E = models.image_output_embed_128(dim=self.args.nz, nc=self.args.nc)
        self.E = self.E.to(self.args.device)

        self.D = models.classifier(dim=self.args.nz, n_class=1, use_bn=False)
        self.D = self.D.to(self.args.device)

        self.C = models.classifier(dim=self.args.nz, n_class=self.args.n_class, use_bn=False)
        self.C = self.C.to(self.args.device)

        self.optim_D = torch.optim.Adam(
                        list(self.E.parameters()) + \
                        list(self.D.parameters()) + \
                        list(self.C.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def save_model(self, path):
        Printer.print('Saving Model on %s ...' % (path))
        state = {
            'enc': self.enc.state_dict(),
            'dec': self.dec.state_dict(),
            'E': self.E.state_dict(),
            'D': self.D.state_dict(),
            'C': self.C.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path, map_location=self.args.device)
        self.enc.load_state_dict(ckpt['enc'])
        self.dec.load_state_dict(ckpt['dec'])
        self.E.load_state_dict(ckpt['E'])
        self.D.load_state_dict(ckpt['D'])
        self.C.load_state_dict(ckpt['C'])

    def save_state(self, path):
        torch.save({
            'epoch': self.epoch,
            'enc': self.enc.state_dict(),
            'dec': self.dec.state_dict(),
            'E': self.E.state_dict(),
            'D': self.D.state_dict(),
            'C': self.C.state_dict(),
            'optim': self.optim.state_dict(),
            'optim_D': self.optim_D.state_dict(),
            'loss': (self.mse, self.l1, self.bce, self.ce),
            'seed': self.args.seed
            }, path)

    def load_state(self, path):
        checkpoint = torch.load(path)
        self.enc.load_state_dict(checkpoint['enc'])
        self.dec.load_state_dict(checkpoint['dec'])
        self.E.load_state_dict(checkpoint['E'])
        self.D.load_state_dict(checkpoint['D'])
        self.C.load_state_dict(checkpoint['C'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.optim_D.load_state_dict(checkpoint['optim_D'])
        self.epoch = checkpoint['epoch']
        self.mse, self.l1, self.bce, self.ce = checkpoint['loss']
        self.args.seed = checkpoint['seed']
        torch.manual_seed(self.args.seed)

    def save_output(self, output, path):
        utils.save_image(output.data, path, normalize=True)

    def zero_grad_G(self):
        self.enc.zero_grad()
        self.dec.zero_grad()
        
    def zero_grad_D(self):
        self.E.zero_grad()
        self.D.zero_grad()
        self.C.zero_grad()

    def set_train(self):
        self.enc.train()
        self.dec.train()
        self.E.train()
        self.D.train()
        self.C.train()

    def set_eval(self):
        self.enc.eval()
        self.dec.eval()
        self.E.eval()
        self.D.eval()
        self.C.eval()

    def train(self, data_loader):
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.set_train()
            record = utils.Record()
            record_G = utils.Record()
            record_D = utils.Record()
            record_C_real = utils.Record() # C1 for ID
            record_C_fake = utils.Record()
            record_C_real_acc = utils.Record() # C1 for ID
            record_C_fake_acc = utils.Record()
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (trace, image, prefix, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
                ID = ID.to(self.args.device)
                bs = image.size(0)

                # train D with real
                self.zero_grad_D()
                real_data = image.to(self.args.device)
                batch_size = real_data.size(0)
                label_real = torch.full((batch_size, 1), self.real_label, dtype=real_data.dtype).to(self.args.device)
                label_fake = torch.full((batch_size, 1), self.fake_label, dtype=real_data.dtype).to(self.args.device)

                embed_real = self.E(real_data)
                output_real = self.D(embed_real)
                errD_real = self.bce(output_real, label_real)
                D_x = output_real.mean().item()

                # train D with fake
                encoded = self.enc(trace)
                noise = torch.randn(bs, self.args.nz).to(self.args.device)
                decoded = self.dec(encoded + 0.05 * noise)
                
                output_fake = self.D(self.E(decoded.detach()))
                errD_fake = self.bce(output_fake, label_fake)
                D_G_z1 = output_fake.mean().item()
                
                errD = errD_real + errD_fake
                
                # train C with real
                pred_real = self.C(embed_real)
                errC_real = self.ce(pred_real, ID)
                (errD_real + errD_fake + errC_real).backward()
                self.optim_D.step()
                record_D.add(errD)
                record_C_real.add(errC_real)
                record_C_real_acc.add(utils.accuracy(pred_real, ID))

                # train G with D and C
                self.zero_grad_G()

                encoded = self.enc(trace)
                noise = torch.randn(bs, self.args.nz).to(self.args.device)
                decoded = self.dec(encoded + 0.05 * noise)

                embed_fake = self.E(decoded)
                output_fake = self.D(embed_fake)
                pred_fake = self.C(embed_fake)

                errG = self.bce(output_fake, label_real)
                errC_fake = self.ce(pred_fake, ID)
                recons_err = self.mse(decoded, image)

                (errG + errC_fake + self.args.lambd * recons_err).backward()
                D_G_z2 = output_fake.mean().item()
                self.optim.step()
                record_G.add(errG)
                record.add(recons_err.item())
                record_C_fake.add(errC_fake)
                record_C_fake_acc.add(utils.accuracy(pred_fake, ID))

            progress.finish()
            utils.clear_progressbar()
            Printer.print('----------------------------------------')
            Printer.print('Epoch: %d' % self.epoch)
            Printer.print('Costs Time: %.2f s' % (time.time() - start_time))
            Printer.print('Recons Loss: %f' % (record.mean()))
            Printer.print('Loss of G: %f' % (record_G.mean()))
            Printer.print('Loss of D: %f' % (record_D.mean()))
            Printer.print('Loss & Acc of C ID real: %f & %f' % (record_C_real.mean(), record_C_real_acc.mean()))
            Printer.print('Loss & Acc of C ID fake: %f & %f' % (record_C_fake.mean(), record_C_fake_acc.mean()))
            Printer.print('D(x) is: %f, D(G(z1)) is: %f, D(G(z2)) is: %f' % (D_x, D_G_z1, D_G_z2))
            self.save_output(decoded, os.path.join(self.args.image_root, ('train_%03d.jpg' % self.epoch)))
            self.save_output(image, os.path.join(self.args.image_root, ('train_%03d_target.jpg' % self.epoch)))
            
    def test(self, data_loader):
        self.set_eval()
        record = utils.Record()
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        with torch.no_grad():
            for i, (trace, image, prefix, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
                encoded = self.enc(trace)
                decoded = self.dec(encoded)                
                recons_err = self.mse(decoded, image)
                record.add(recons_err.item())

                if i == 0:
                    self.save_output(decoded, os.path.join(self.args.image_root, ('test_%03d.jpg' % self.epoch)))
                    self.save_output(image, os.path.join(self.args.image_root, ('test_%03d_target.jpg' % self.epoch)))

            progress.finish()
            utils.clear_progressbar()
            Printer.print('----------------------------------------')
            Printer.print('Test.')
            Printer.print('Costs Time: %.2f s' % (time.time() - start_time))
            Printer.print('Recons Loss: %f' % (record.mean()))

if __name__ == '__main__':
    args = Params().parse()
    assert args.dataset == 'CelebA'

    args.trace_c = 6
    args.trace_w = 256
    args.nz = 128

    args.image_root = os.path.join(args.output_root, args.exp_name, 'image')
    args.ckpt_root = os.path.join(args.output_root, args.exp_name, 'ckpt')
    Printer.output_file = os.path.join(args.output_root, args.exp_name, 'output.out')

    Printer.print(f'Experiment Name: { args.exp_name }')
    args.seed = random.randint(1, 10000)
    Printer.print('Manual Seed: %d' % args.seed)
    torch.manual_seed(args.seed)
    
    loader = DataLoader(args)
    train_dataset = CelebaDataset(
                    img_dir=args.data_path[args.dataset]['media'], 
                    npz_dir=args.data_path[args.dataset][args.side],
                    ID_path=args.data_path[args.dataset]['ID_path'],
                    split=args.data_path[args.dataset]['split'][0],
                    trace_c=args.trace_c,
                    trace_w=args.trace_w,
                    image_size=args.image_size,
                    side=args.side
                )
    args.n_class = train_dataset.ID_cnt

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

    engine = ImageEngine(args)

    train_loader = loader.get_loader(train_dataset)
    test_loader = loader.get_loader(test_dataset, shuffle=False)

    if os.path.exists(args.output_root + args.exp_name):
        ans = input(f'Experiment folder "{ args.exp_name }" already exist, do you want to continue training or overwrite the result? (continue/overwrite/ctrl+c) ')
        if ans.lower() == 'continue':
            print("Continue training")
            engine.load_state(os.path.join(args.output_root, 'temp_state'))
        elif ans.lower() == 'overwrite':
            print("Overwrite previous experiment result")
        else:
            print("Unknown input. Program terminate")
            exit(1)
    else:
        os.mkdir(os.path.join(args.output_root, args.exp_name))
        utils.make_path(args.image_root)
        utils.make_path(args.ckpt_root)

    for i in range(engine.epoch, args.num_epoch):
        engine.train(train_loader)
        if i % args.test_freq == 0:
            engine.test(test_loader)
            engine.save_model((args.ckpt_root + '/%03d.pth') % (i + 1))
        engine.save_state(os.path.join(args.output_root, 'temp_state.pth'))
    engine.save_model((args.ckpt_root + '/final.pth'))
