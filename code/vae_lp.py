import time
import progressbar

import torch
import torch.nn as nn

from models2 import ImageEncoder, ImageDecoder
import utils


class VAE_LP:
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.l1 = nn.L1Loss().cuda()
        self.l2 = nn.MSELoss().cuda()
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model & Optimizer...')
        self.ImageEncoder = ImageEncoder(nc=self.args['nc'], dim=self.args['vae_dim'])
        self.ImageEncoder = torch.nn.DataParallel(self.ImageEncoder).cuda()
        
        self.Decoder = ImageDecoder(nc=self.args['nc'], dim=self.args['vae_dim'])
        self.Decoder = torch.nn.DataParallel(self.Decoder).cuda()
        
        self.optimizer = torch.optim.Adam(
                            list(self.ImageEncoder.module.parameters()) + \
                            list(self.Decoder.module.parameters()),
                            lr=self.args['vae_lr'],
                            betas=(self.args['beta1'], 0.999)
                            )

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.ImageEncoder.module.load_state_dict(ckpt['ImageEncoder'])
        self.Decoder.module.load_state_dict(ckpt['Decoder'])

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'ImageEncoder': self.ImageEncoder.module.state_dict(),
            'Decoder': self.Decoder.module.state_dict()
        }
        torch.save(state, path)

    def set_train(self):
        self.ImageEncoder.train()
        self.Decoder.train()

    def set_eval(self):
        self.ImageEncoder.eval()
        self.Decoder.eval()

    def zero_grad(self):
        self.ImageEncoder.zero_grad()
        self.Decoder.zero_grad()

    def train(self, data_loader):
        print('Training...')
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.set_train()
            record_image = utils.Record()
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, image in enumerate(data_loader):
                progress.update(i + 1)
                image = image.cuda()
                self.zero_grad()
                mu, log_var, encoded = self.ImageEncoder(image)
                image2image = self.Decoder(encoded)

                recons_loss = self.l1(image2image, image)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0) #??
                kld_weight = self.args['kld_weight']
                loss = recons_loss + kld_weight * kld_loss

                loss.backward()
                self.optimizer.step()

                record_image.add(loss)
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Epoch: %d' % self.epoch)
            print('Costs time: %.2fs' % (time.time() - start_time))
            print('Loss of Image to Image: %f' % (record_image.mean()))
            print('----------------------------------------')
            utils.save_image(image.data, ('%s/image/train/%03d_target.jpg' % (self.args['vae_dir'], self.epoch)), normalize=True)
            utils.save_image(image2image.data, ('%s/image/train/%03d_recon.jpg' % (self.args['vae_dir'], self.epoch)), normalize=True)

    def test(self, data_loader):
        print('Testing...')
        with torch.no_grad():
            self.set_eval()
            record_image = utils.Record()
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, image in enumerate(data_loader):
                progress.update(i + 1)
                image = image.cuda()

                mu, log_var, encoded = self.ImageEncoder(image)
                image2image = self.Decoder(encoded)

                recons_loss = self.l1(image2image, image)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                kld_weight = self.args['kld_weight']
                loss = recons_loss + kld_weight * kld_loss

                record_image.add(loss)
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Test at Epoch %d' % self.epoch)
            print('Costs time: %.2fs' % (time.time() - start_time))
            print('Loss of Image to Image: %f' % (record_image.mean()))
            print('----------------------------------------')
            utils.save_image(image.data, ('%s/image/test/%03d_target.jpg' % (self.args['vae_dir'], self.epoch)), normalize=True)
            utils.save_image(image2image.data, ('%s/image/test/%03d_recon.jpg' % (self.args['vae_dir'], self.epoch)), normalize=True)

    def inference(self, x):
        with torch.no_grad():
            self.ImageEncoder.eval()
            self.Decoder.eval()
            trace_embed = self.ImageEncoder(x)
            recov_image = self.Decoder(trace_embed)
        return recov_image

    def generate(self):
        with torch.no_grad():
            self.Decoder.eval()
            random_vec = torch.rand(self.args['batch_size'], self.args['vae_dim'])
            for i in range(50):
                for j in range(128):
                    random_vec[i][j] *= 100 - 50
            print(random_vec)
            output = self.Decoder(random_vec)
        return output
