#from .networks import RegisterNetwork
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torch.nn.utils import spectral_norm
#from .spectral_norm import spectral_norm

_NETWORKS = {}
_ENCODER_NETWORKS = {}
_DECODER_NETWORKS = {}
_PREDICTOR_NETWORKS = {}

class RegisterNetwork(object):
    def __init__(self, name, encoder=None, decoder=None, predictor=None):
        self.network = name
        self.encoder = encoder or name
        self.decoder = decoder or name
        self.predictor = predictor or name

    def __call__(self, cls):
        _NETWORKS[self.network] = cls
        _ENCODER_NETWORKS[self.encoder] = cls
        _DECODER_NETWORKS[self.decoder] = cls
        _PREDICTOR_NETWORKS[self.decoder] = cls
        return cls

@RegisterNetwork('identity')
class Identity(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        return x

class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True, device=None, dtype=None):
        nn.BatchNorm1d.__init__(self, num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)
        
    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.uniform_(self.weight, -1., 1)
            init.zeros_(self.bias)

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True, device=None, dtype=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #nn.BatchNorm2d.__init__(self, num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)
        nn.BatchNorm2d.__init__(self, num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.uniform_(self.weight, -1., 1)
            init.zeros_(self.bias)


class ResidualBlock(nn.Module):
    """Basic 2D convolutional block with residual connections. From WRN paper.

    Args:
        in_channels (int): Channels of the input image.
        out_features (int): Output channels.
        stride: The stride for convolution.
        kernel_size: The kernel size for for the convolutional layers.
        activation: An activation function for every layer. Defaults to `relu`.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=F.relu, bn=True, sn=False):
        nn.Module.__init__(self)
        padding = kernel_size // 2
        SN = spectral_norm if sn else lambda x: x
        self.layer_1 = SN(nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding))
        self.layer_2 = SN(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.shortcut = SN(nn.Conv2d(in_channels, out_channels, 1, stride, 0))# if stride > 1 or in_channels != out_channels else nn.Identity()
        self.bn_1 = BatchNorm2d(in_channels) if bn else nn.Identity()
        self.bn_2 = BatchNorm2d(in_channels) if bn else nn.Identity()
        self.activation = activation

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.bn_1(x)
        x = self.activation(x)
        x = self.layer_1(x)
        x = self.bn_2(x)
        x = self.activation(x)
        x = self.layer_2(x)
        return x + residual
    
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

    
def BottleneckBlock(*args, **kwargs):
    #torch.jit.script
    return (BottleneckBlockModule(*args, **kwargs))
    
class BottleneckBlockModule(nn.Module):
    """Basic 2D convolutional block with residual connections. From WRN paper.

    Args:
        in_channels (int): Channels of the input image.
        out_features (int): Output channels.
        stride: The stride for convolution.
        kernel_size: The kernel size for for the convolutional layers.
        activation: An activation function for every layer. Defaults to `relu`.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, stride=1, activation=F.gelu, bn=True, weight_scale=1., residual=True, resample=None, zero_bias=True, groups=1, SE=True, SN=True, both3x3=True):
        nn.Module.__init__(self)
        self.residual = residual
        padding = kernel_size // 2
        if mid_channels is None:
            mid_channels = out_channels // 4
            
        self.resample = nn.Identity()
        if resample == 'up':
            self.resample = nn.Upsample(scale_factor=2)
        elif resample == 'down' or stride == 2:
            stride = 1
            self.resample = nn.AvgPool2d(2, 2)
            
        SN = spectral_norm if SN else lambda x: x
        self.layer_1 = SN(nn.Conv2d(in_channels, mid_channels, 1, 1))
        self.layer_2 = SN(nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=groups))
        self.layer_3 = SN(nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=groups)) if both3x3 else nn.Identity()
        self.layer_4 = SN(nn.Conv2d(mid_channels, out_channels, 1, 1))
        self.layer_4.weight.data *= weight_scale
        
        if zero_bias:
            self.layer_1.bias.data *= 0.
            self.layer_2.bias.data *= 0.
            self.layer_3.bias.data *= 0.
            self.layer_4.bias.data *= 0.
        
        self.bn_1 = BatchNorm2d(in_channels) if bn else nn.Identity()
        self.bn_2 = BatchNorm2d(mid_channels) if bn else nn.Identity()
        self.bn_3 = BatchNorm2d(mid_channels) if bn else nn.Identity()
        self.bn_4 = BatchNorm2d(mid_channels) if bn else nn.Identity()
        self.dim_adjust = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, 1)
        self.activation = activation
        self.se = SE_Block(out_channels) if SE else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.bn_1(x)
        x = self.activation(x)
        x = self.layer_1(x)
        x = self.bn_2(x)
        x = self.activation(x)
        x = self.layer_2(x)
        x = self.bn_3(x)
        x = self.activation(x)
        x = self.layer_3(x)
        x = self.bn_4(x)
        x = self.activation(x)
        x = self.layer_4(x)
        x = self.se(x)
        if self.residual:
            residual = self.dim_adjust(residual)
            x = x + residual
        return self.resample(x)

class UpscaleBlock(ResidualBlock):
    """Basic 2D transposed convolutional block with residual connections. Upscales resolution.

    Args:
        in_channels (int): Channels of the input image.
        out_features (int): Output channels.
        stride: The stride for convolution.
        kernel_size: The kernel size for for the convolutional layers.
        activation: An activation function for every layer. Defaults to `relu`.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=F.relu, bn=True, sn=False, transpose=False): 
        nn.Module.__init__(self)
        SN = spectral_norm if sn else lambda x: x
        if stride > 1 or transpose:
            self.layer_1 = SN(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, 1, output_padding=stride - 1))
        else:
            self.layer_1 = SN(nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1))

        if transpose:
            self.layer_2 = SN(nn.ConvTranspose2d(out_channels, out_channels, kernel_size, 1, 1))
        else:
            self.layer_2 = SN(nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1))
            
        self.shortcut = SN(nn.Conv2d(in_channels, out_channels, 1, 1))# if stride > 1 or in_channels != out_channels else nn.Identity()
        self.upsample = nn.Upsample(scale_factor=stride) if stride > 1 or in_channels != out_channels else nn.Identity()
        self.bn_1 = BatchNorm2d(in_channels) if bn else nn.Identity()
        self.bn_2 = BatchNorm2d(out_channels) if bn else nn.Identity()
        self.activation = activation

    def forward(self, x):
        residual = self.shortcut(x)
        residual = self.upsample(residual)
        x = self.bn_1(x)
        x = self.activation(x)
        x = self.layer_1(x)
        x = self.bn_2(x)
        x = self.activation(x)
        x = self.layer_2(x)
        return x + residual


class DenseBlock(ResidualBlock):
    """Basic dense block with residual connections.

    Args:
        in_features (int): Features of the input tensor.
        out_features (int): Output features.
        stride: The stride for convolution.
        kernel_size: The kernel size for for the convolutional layers.
        activation: An activation function for every layer. Defaults to `relu`.
    """
    def __init__(self, in_features, out_features, activation=F.relu, residual=False, bn=True):
        nn.Module.__init__(self)
        self.layer_1 = (nn.Linear(in_features, in_features))
        self.layer_2 = (nn.Linear(in_features, out_features))
        self.shortcut = (nn.Linear(in_features, out_features)) if in_features != out_features else nn.Identity()
        self.bn_1 = BatchNorm1d(in_features) if bn else nn.Identity()
        self.bn_2 = BatchNorm1d(in_features) if bn else nn.Identity()
        self.activation = activation
        self.residual = residual

    def forward(self, x):
        if not self.residual:
            x = self.activation(x)
            x = self.layer_1(x)
            return x

        residual = self.shortcut(x)
        x = self.bn_1(x)
        x = self.activation(x)
        x = self.layer_1(x)
        x = self.bn_2(x)
        x = self.activation(x)
        x = self.layer_2(x)
        return x + residual


class WRNLevel(nn.Module):
    """Wide ResNet level that downscales by a factor of 2.

    Args:
        in_channels (int): Channels of the input image.
        out_channels (int): Output channels.
        blocks: The number of internal residual blocks.
        kernel_size: The kernel size for for the convolutional layers.
        activation: An activation function for every layer. Defaults to `relu`.
    """
    def __init__(self, in_channels, out_channels, blocks=4, kernel_size=3, activation=F.relu, bn=True, stride=2, blocktype='wrn'):
        nn.Module.__init__(self)
        Block = BottleneckBlockModule if blocktype == 'bottle' else ResidualBlock
        self.resize_block = Block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, activation=activation, bn=bn)
        self.blocks = nn.ModuleList(
            [Block(out_channels, out_channels, kernel_size=kernel_size, stride=1, activation=activation, bn=bn) for i in range(blocks - 1)])

    def forward(self, x):
        x = self.resize_block(x)
        for block in self.blocks:
            x = block(x)
        return x


class WRNUpscaleLevel(WRNLevel):
    """Wide ResNet upscaling level that upscales by a factor of 2.

    Args:
        in_channels (int): Channels of the input image.
        out_channels (int): Output channels.
        blocks: The number of internal residual blocks.
        kernel_size: The kernel size for for the convolutional layers.
        activation: An activation function for every layer. Defaults to `relu`.
    """
    def __init__(self, in_channels, out_channels, blocks=4, kernel_size=3, activation=F.relu, stride=2, bn=True, transpose=False):
        nn.Module.__init__(self)
        self.resize_block = UpscaleBlock(in_channels, out_channels, kernel_size, stride, activation, transpose=transpose)
        self.blocks = nn.ModuleList(
            [UpscaleBlock(in_channels, in_channels, kernel_size, 1, activation, bn=bn, transpose=transpose) for i in range(blocks - 1)])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.resize_block(x)
        return x


@RegisterNetwork('wrn_conv', decoder='wrn_conv_inverse')
class WideResNet(nn.Module):
    """A full 2D Wide residual network.

    Note::
        The output will still be 2D (images) use `FlatWideResNet` for flattened outputs.

    Args:
        channels (int): Channels of the input image.
        size (int): The size factor of the wrn.
        levels (int): The number of downscaling levels.
        blocks_per_level: The number of internal residual blocks per downscaling level.
        kernel_size: The kernel size for for the convolutional layers.
        activation: An activation function for every layer. Defaults to `relu`.
    """
    def __init__(self, channels, size=2, levels=4, blocks_per_level=4, kernel_size=3, activation=F.relu,
                 out_features=None, shape=None, starter_net=None, bn=True, blocktype='wrn'):
        nn.Module.__init__(self)
        if starter_net:
            shape = starter_net.shape
            size = starter_net.size
            self.initial_conv = starter_net.initial_conv
            levels += len(starter_net.levels)
        else:
            self.initial_conv = nn.Conv2d(channels, 16, kernel_size, 1, kernel_size // 2)
        in_channels, out_channels = 16, size * 16
        self.levels = nn.ModuleList()
        self.shape = shape
        self.size = size
        for i in range(levels):
            if starter_net and i < len(starter_net.levels):
                self.levels.append(starter_net.levels[i])
            else:
                self.levels.append(WRNLevel(in_channels, out_channels, blocks_per_level, kernel_size, activation, bn=bn, stride=(1 if i == 0 else 2), blocktype=blocktype))
            in_channels, out_channels = out_channels, out_channels * 2
        self.output_conv = nn.Conv2d(in_channels, out_features, 1, 1) if out_features else Identity()

    def forward(self, x):
        x = self.initial_conv(x)
        for level in self.levels:
            x = level(x)
        x = self.output_conv(x)
        return x


@RegisterNetwork('wrn_conv_upscale', encoder='wrn_conv_inverse', decoder='wrn_conv')
class WideResNetUpscaling(nn.Module):
    """A transposed (upscaling) 2D Wide residual network.

    Note::
        The output will still be 2D (images) use `FlatWideResNet` for flattened outputs.

    Args:
        channels (int): Channels of the input image.
        size (int): The size factor of the wrn.
        levels (int): The number of upscaling levels.
        blocks_per_level: The number of internal residual blocks per downscaling level.
        kernel_size: The kernel size for for the convolutional layers.
        activation: An activation function for every layer. Defaults to `relu`.
    """
    def __init__(self, channels, size=2, levels=4, blocks_per_level=4, kernel_size=3, activation=F.relu,
                 in_features=None, shape=None, bn=True, transpose=False):
        nn.Module.__init__(self)
        self.output_conv = nn.Conv2d(size * 16, channels, kernel_size, 1, kernel_size // 2)
        in_channels = (size * 16) * (2 ** (levels - 1))
        out_channels = in_channels

        self.initial_conv = nn.Conv2d(in_features, in_channels, 1, 1) if in_features else Identity()
        self.levels = nn.ModuleList()
        for i in range(levels):
            self.levels.append(WRNUpscaleLevel(in_channels, out_channels, blocks_per_level, kernel_size, activation, bn=bn, transpose=transpose, stride=(1 if i == (levels - 1) else 2)))
            in_channels, out_channels = out_channels, out_channels // 2

    def forward(self, x):
        x = self.initial_conv(x)
        for level in self.levels:
            x = level(x)
        x = self.output_conv(x)
        return x


@RegisterNetwork('input_wrn', decoder='input_wrn_inverse')
class InputWideResNet(nn.Module):
    """Wide ResNet with no downscaling.

    Args:
        channels (int): Channels of the input image.
        blocks: The number of internal residual blocks.
        kernel_size: The kernel size for for the convolutional layers.
        activation: An activation function for every layer. Defaults to `relu`.
    """
    def __init__(self, channels, out_features=None, shape=None, units=128, blocks=4, kernel_size=3, activation=F.relu, bn=True):
        nn.Module.__init__(self)
        self.resize_block = ResidualBlock(channels, units, kernel_size, 1, activation)
        self.blocks = nn.ModuleList(
            [ResidualBlock(units, units, kernel_size, 1, activation, bn=bn) for i in range(blocks - 1)])
        self.output_layer = nn.Conv2d(units, out_features, 1, 1)

    def forward(self, x):
        x = self.resize_block(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


@RegisterNetwork('output_wrn', encoder='output_wrn_inverse')
class OutputWideResNet(nn.Module):
    """Transposed Wide ResNet with no upscaling.

    Args:
        channels (int): Channels of the input image.
        blocks: The number of internal residual blocks.
        kernel_size: The kernel size for for the convolutional layers.
        activation: An activation function for every layer. Defaults to `relu`.
    """
    def __init__(self, channels, in_features=None, shape=None, units=128, blocks=4, kernel_size=3, activation=F.relu, bn=True):
        nn.Module.__init__(self)
        self.resize_block = UpscaleBlock(in_features, units, kernel_size, 1, activation)
        self.blocks = nn.ModuleList([UpscaleBlock(units, units, kernel_size, 1, activation, bn=bn) for i in range(blocks - 1)])
        self.output_layer = nn.ConvTranspose2d(units, channels, 1, 1)

    def forward(self, x):
        x = self.resize_block(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


def wrnsize(in_shape, size, levels):
    """Utility functino to calculate the output channels and shape given a WRN spec."""
    out_shape, out_channels = in_shape, size * 16
    for i in range(levels - 1):
        out_shape = (out_shape[0] // 2, out_shape[1] // 2)
        out_channels = out_channels * 2
    return (out_channels,) + out_shape, out_shape[0] * out_shape[1] * out_channels


class GlobalPool2d(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        
    def forward(self, x):
        return torch.mean(x, dim=(-1, -2))

@RegisterNetwork('wrn', decoder='wrn_inverse')
class FlatWideResNet(nn.Module):
    def __init__(self, channels, shape=(32, 32), out_features=None, size=2, levels=4, blocks_per_level=4, kernel_size=3,
                 activation=F.gelu, bn=True, dense_blocks=0, dense_units=1000, label_dims=0, avg_pool=False, starter_net=None, blocktype='wrn'):
        nn.Module.__init__(self)
        if starter_net:
            shape = starter_net.shape
            size = starter_net.size
        out_features = dense_units if out_features is None else out_features
        self.wrn = WideResNet(channels, size, levels, blocks_per_level, kernel_size, activation, starter_net=starter_net, bn=bn, blocktype=blocktype)
        self.flatten = nn.Flatten()
        self.activation = activation
        self.label_dims = label_dims

        # Get the number of input features for the dense layers
        levels = len(self.wrn.levels)
        _, features = wrnsize(shape, size, levels)
        
        if avg_pool:
            features = 8 * size * (2 ** levels)
            self.flatten = GlobalPool2d()

        self.dense_blocks = nn.ModuleList(
            [DenseBlock(dense_units, dense_units, activation, residual=True, bn=False) for i in range(dense_blocks)])

        if dense_blocks:
            self.dense_blocks.insert(0, nn.Linear(features + label_dims, dense_units))
            self.output_layer = nn.Linear(dense_units, out_features)
        else:
            self.output_layer = nn.Linear(features, out_features)

    #@torch.cuda.amp.autocast()
    def forward(self, x, y=torch.tensor(0.)):
        x = self.wrn(x)
        x = self.flatten(x)
        if self.label_dims:
            x = torch.cat([x, y], dim=1)
        for block in self.dense_blocks:
            x = block(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x.float()

@RegisterNetwork('wrn_upscale', encoder='wrn_inverse', decoder='wrn')
class FlatWideResNetUpscaling(nn.Module):
    def __init__(self, channels, shape=(32, 32), in_features=None, size=2, levels=4, blocks_per_level=4, kernel_size=3,
                 activation=F.gelu, dense_blocks=0, bn=True, avg_pool=False, dense_units=1000, transpose=False, model="vae", evaluate=False):
        
        nn.Module.__init__(self)
        in_features = dense_units if in_features is None else in_features
        conv_shape, features = wrnsize(shape, size, levels)
        self.conv_shape = conv_shape
        self.activation = activation
        self.model = model
        self.dense_blocks = nn.ModuleList(
            [DenseBlock(dense_units, dense_units, activation, residual=True, bn=False) for i in range(dense_blocks)])
        if dense_blocks:
            self.input_layer = nn.Linear(in_features, dense_units)
            self.dense_blocks.append(nn.Linear(dense_units, features))
        else:
            self.input_layer = nn.Linear(in_features, features)

        self.log_sigma = 0
        if self.model == 'sigma_vae':
            ## Sigma VAE
            #self.log_sigma = torch.nn.Parameter(torch.full((1,), 0)[0], requires_grad=True)
            if evaluate:
                self.log_sigma = torch.nn.Parameter(torch.rand(1, device='cuda'), requires_grad = False)
            else:
                self.log_sigma = torch.nn.Parameter(torch.rand(1, device='cuda'), requires_grad = True)

        self.wrn = WideResNetUpscaling(channels, size, levels, blocks_per_level, kernel_size, activation, bn=bn, transpose=transpose)

    #@torch.cuda.amp.autocast()
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.dense_blocks:
            x = block(x)
        x = self.activation(x)
        x = torch.reshape(x, (-1,) + self.conv_shape)

        x = self.wrn(x)
        return x.float()
