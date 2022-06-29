import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
from model.vgg19_encoder import vgg19_encoder
from model.vgg19_decoder import vgg19_decoder


def _apply_weight_encoder(original_state_dict_path, encoder_to_save_path):
    vgg_enc = vgg19_encoder()
    original_state_dict = torch.load(original_state_dict_path)    
    original_weights = list(original_state_dict.values())

    for layer in vgg_enc.children():
        if isinstance(layer, nn.Conv2d):
            layer.weight.data = original_weights.pop(0)
            layer.bias.data = original_weights.pop(0)

    torch.save(vgg_enc.state_dict(), encoder_to_save_path)


def _apply_weight_decoder(encoder_state_dict_path, decoder_to_save_path):
    vgg_dec = vgg19_decoder()
    encoder_state_dict = torch.load(encoder_state_dict_path)    
    encoder_weights = list(encoder_state_dict.values())
    encoder_weights = encoder_weights[::-1]

    for layer in vgg_dec.children():
        if isinstance(layer, nn.Conv2d):
            layer.weight.data = encoder_weights.pop(0)
            layer.bias.data = encoder_weights.pop(0)

    torch.save(vgg_dec.state_dict(), decoder_to_save_path)

if __name__ == '__main__':
    WEIGHT_DIR = os.path.join(ROOT_DIR, 'model', 'weights')
    original_state_dict_path = os.path.join(WEIGHT_DIR, 'vgg19_original.pt')
    encoder_state_dict_path = os.path.join(WEIGHT_DIR, 'vgg19_encoder.pt')

    # encoder_to_save_path = os.path.join(WEIGHT_DIR, 'vgg19_encoder.pt')
    decoder_to_save_path = os.path.join(WEIGHT_DIR, 'vgg19_decoder.pt')
    _apply_weight_decoder(encoder_state_dict_path=encoder_state_dict_path,
                          decoder_to_save_path=decoder_to_save_path)
    # _apply_weight_encoder(original_state_dict_path=original_state_dict_path,
    #                       encoder_to_save_path=encoder_to_save_path)
    