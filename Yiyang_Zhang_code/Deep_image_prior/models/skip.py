import torch
import torch.nn as nn
from .common import *

def skip(
        in_channels, out_channels=3, 
        channels_down=[16, 32, 64, 128, 128], channels_up=[16, 32, 64, 128, 128], channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        use_sigmoid=True, use_bias=True, 
        padding='zero', up_mode='nearest', down_mode='stride', activation='LeakyReLU', 
        use_1x1_up=True):
    """Assembles an encoder-decoder network with skip connections.

    Arguments:
        activation: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        padding (string): zero|reflection (default: 'zero')
        up_mode (string): 'nearest|bilinear' (default: 'nearest')
        down_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(channels_down) == len(channels_up) == len(channels_skip)

    scales = len(channels_down) 

    up_mode = [up_mode] * scales if not isinstance(up_mode, (list, tuple)) else up_mode
    down_mode = [down_mode] * scales if not isinstance(down_mode, (list, tuple)) else down_mode
    filter_size_down = [filter_size_down] * scales if not isinstance(filter_size_down, (list, tuple)) else filter_size_down
    filter_size_up = [filter_size_up] * scales if not isinstance(filter_size_up, (list, tuple)) else filter_size_up

    final_scale = scales - 1 

    network = nn.Sequential()
    current_block = network

    current_depth = in_channels
    for scale in range(scales):

        deeper_block = nn.Sequential()
        skip_block = nn.Sequential()

        if channels_skip[scale] != 0:
            current_block.add_module(f'concat_{scale}', Concat(1, skip_block, deeper_block))
        else:
            current_block.add_module(f'deeper_{scale}', deeper_block)
        
        current_block.add_module(f'batch_norm_{scale}', bn(channels_skip[scale] + (channels_up[scale + 1] if scale < final_scale else channels_down[scale])))

        if channels_skip[scale] != 0:
            skip_block.add_module(f'skip_conv_{scale}', conv(current_depth, channels_skip[scale], filter_skip_size, bias=use_bias, pad=padding))
            skip_block.add_module(f'skip_bn_{scale}', bn(channels_skip[scale]))
            skip_block.add_module(f'skip_act_{scale}', act(activation))
            
        deeper_block.add_module(f'down_conv1_{scale}', conv(current_depth, channels_down[scale], filter_size_down[scale], 2, bias=use_bias, pad=padding, downsample_mode=down_mode[scale]))
        deeper_block.add_module(f'down_bn1_{scale}', bn(channels_down[scale]))
        deeper_block.add_module(f'down_act1_{scale}', act(activation))

        deeper_block.add_module(f'down_conv2_{scale}', conv(channels_down[scale], channels_down[scale], filter_size_down[scale], bias=use_bias, pad=padding))
        deeper_block.add_module(f'down_bn2_{scale}', bn(channels_down[scale]))
        deeper_block.add_module(f'down_act2_{scale}', act(activation))

        inner_deeper_block = nn.Sequential()

        if scale == final_scale:
            inner_channels = channels_down[scale]
        else:
            deeper_block.add_module(f'inner_deeper_block_{scale}', inner_deeper_block)
            inner_channels = channels_up[scale + 1]

        deeper_block.add_module(f'upsample_{scale}', nn.Upsample(scale_factor=2, mode=up_mode[scale]))

        current_block.add_module(f'up_conv_{scale}', conv(channels_skip[scale] + inner_channels, channels_up[scale], filter_size_up[scale], 1, bias=use_bias, pad=padding))
        current_block.add_module(f'up_bn_{scale}', bn(channels_up[scale]))
        current_block.add_module(f'up_act_{scale}', act(activation))

        if use_1x1_up:
            current_block.add_module(f'conv_1x1_{scale}', conv(channels_up[scale], channels_up[scale], 1, bias=use_bias, pad=padding))
            current_block.add_module(f'bn_1x1_{scale}', bn(channels_up[scale]))
            current_block.add_module(f'act_1x1_{scale}', act(activation))

        current_depth = channels_down[scale]
        current_block = inner_deeper_block

    network.add_module('final_conv', conv(channels_up[0], out_channels, 1, bias=use_bias, pad=padding))
    if use_sigmoid:
        network.add_module('sigmoid', nn.Sigmoid())

    return network