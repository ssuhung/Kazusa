import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class ImageDecoder(nn.Module):
    def __init__(self, dim, nc, padding_type='reflect', norm_layer=nn.InstanceNorm2d, #nn.BatchNorm2d
                use_dropout=False, use_bias=False):
        super(ImageDecoder, self).__init__()
        self.dim = dim
        self.nc = nc
        self.net_part1 = nn.Sequential(
            # state size. (1) x 1 x 1
            nn.ConvTranspose2d(self.dim, self.dim, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(self.dim), #nn.BatchNorm2d(self.dim), 
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 4 x 4
        )
        self.net_part2 = nn.Sequential(
            # state size. (1) x 4 x 4
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.dim), #nn.BatchNorm2d(self.dim), 
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 8 x 8
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.dim), #nn.BatchNorm2d(self.dim), 
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 16 x 16
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.dim), #nn.BatchNorm2d(self.dim), 
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 32 x 32
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.dim), #nn.BatchNorm2d(self.dim), 
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 64 x 64
            nn.ConvTranspose2d(self.dim, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )
        self.apply(weights_init)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        tmp = self.net_part1(input)
        output = self.net_part2(tmp)
        return output

class ImageEncoder(nn.Module):
    def __init__(self, nc, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ImageEncoder, self).__init__()
        self.nc = nc
        self.dim = dim
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(self.nc, self.dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 64 x 64
            nn.Conv2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 32 x 32
            nn.Conv2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 16 x 16
            nn.Conv2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 8 x 8
            nn.Conv2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 4 x 4
            # nn.Conv2d(self.dim, self.dim, 4, 1, 0, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim) x 1 x 1
        )
        self.fc_mu = nn.Linear(dim*16, dim)
        self.fc_var = nn.Linear(dim*16, dim)
        self.apply(weights_init)

    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        output = self.main(input)
        output = torch.flatten(output, start_dim=1)
        mu = self.fc_mu(output)
        log_var = self.fc_var(output)
        output = self.reparameterize(mu, log_var)

        return mu, log_var, output


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Generator(nn.Module):
    def __init__(self, nc, dim):
        super(Generator, self).__init__()

        self.nc = nc
        self.dim = dim
        # 128
        self.net1 = nn.Sequential(
                    nn.Conv2d(nc, dim, 4, 2, 1)
                    )
        self.net2 = nn.Sequential(
                    nn.Conv2d(dim, dim * 2, 4, 2, 1),
                    nn.BatchNorm2d(dim * 2)
                    )
        self.net3 = nn.Sequential(
                    nn.Conv2d(dim * 2, dim * 4, 4, 2, 1),
                    nn.BatchNorm2d(dim * 4)
                    )
        self.net4 = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 8, 4, 2, 1),
                    nn.BatchNorm2d(dim * 8)
                    )
        self.net5 = nn.Sequential(
                    nn.Conv2d(dim * 8, dim * 8, 4, 2, 1),
                    nn.BatchNorm2d(dim * 8)
                    )
        self.net6 = nn.Sequential(
                    nn.Conv2d(dim * 8, dim * 8, 4, 2, 1),
                    nn.BatchNorm2d(dim * 8)
                    )
        self.net7 = nn.Sequential(
                    nn.Conv2d(dim * 8, dim * 8, 4, 2, 1),
                    #nn.BatchNorm2d(dim * 8)
                    )

        self.dnet1 = nn.Sequential(
                    nn.ConvTranspose2d(dim * 8 , dim * 8, 4, 2, 1),
                    nn.BatchNorm2d(dim * 8)
                    )
        self.dnet2 = nn.Sequential(
                    nn.ConvTranspose2d(dim * 8 , dim * 8, 4, 2, 1),
                    nn.BatchNorm2d(dim * 8)
                    )
        self.dnet3 = nn.Sequential(
                    nn.ConvTranspose2d(dim * 8 , dim * 8, 4, 2, 1),
                    nn.BatchNorm2d(dim * 8)
                    )
        self.dnet4 = nn.Sequential(
                    nn.ConvTranspose2d(dim * 8 , dim * 4, 4, 2, 1),
                    nn.BatchNorm2d(dim * 4)
                    )
        self.dnet5 = nn.Sequential(
                    nn.ConvTranspose2d(dim * 4 , dim * 2, 4, 2, 1),
                    nn.BatchNorm2d(dim * 2)
                    )
        self.dnet6 = nn.Sequential(
                    nn.ConvTranspose2d(dim * 2 , dim, 4, 2, 1),
                    nn.BatchNorm2d(dim)
                    )
        self.dnet7 = nn.Sequential(
                    nn.ConvTranspose2d(dim , nc, 4, 2, 1)
                    )

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # input is (nc) x 128 x 128
        e1 = self.net1(input)
        # state size is (dim) x 64 x 64
        e2 = self.net2(self.leaky_relu(e1))
        # state size is (dim x 2) x 32 x 32
        e3 = self.net3(self.leaky_relu(e2))
        # state size is (dim x 4) x 16 x 16
        e4 = self.net4(self.leaky_relu(e3))
        # state size is (dim x 8) x 8 x 8
        e5 = self.net5(self.leaky_relu(e4))
        # state size is (dim x 8) x 4 x 4
        e6 = self.net6(self.leaky_relu(e5))
        # state size is (dim x 8) x 2 x 2
        e7 = self.net7(self.leaky_relu(e6))
        # state size is (dim x 8) x 1 x 1
        d1 = self.dropout(self.dnet1(self.relu(e7)))
        # state size is (dim x 8) x 2 x 2
        d2 = self.dropout(self.dnet2(self.relu(d1)))
        # state size is (dim x 8) x 4 x 4
        d3 = self.dropout(self.dnet3(self.relu(d2)))
        # state size is (dim x 8) x 8 x 8
        d4 = self.dnet4(self.relu(d3))
        # state size is (dim x 4) x 16 x 16
        d5 = self.dnet5(self.relu(d4))
        # state size is (dim x 2) x 32 x 32
        d6 = self.dnet6(self.relu(d5))
        # state size is (dim) x 64 x 64
        d7 = self.dnet7(self.relu(d6))
        # state size is (nc) x 128 x 128
        output = self.tanh(d7)
        return output

class Discriminator(nn.Module):
    def __init__(self, nc, dim):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.dim = dim
        self.main = nn.Sequential(
            # state size. (nc) x 128 x 128
            nn.Conv2d(self.nc, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim) x 64 x 64
            nn.Conv2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim) x 32 x 32
            nn.Conv2d(self.dim, self.dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim*2) x 16 x 16
            nn.Conv2d(self.dim * 2, self.dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim*4) x 8 x 8
            nn.Conv2d(self.dim * 4, self.dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim*8) x 4 x 4
            nn.Conv2d(self.dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1)
