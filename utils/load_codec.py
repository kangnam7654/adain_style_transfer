import torch

from model.vgg19_encoder import vgg19_encoder
from model.vgg19_decoder import vgg19_decoder


def load_codec(device, encoder_pt_path, decoder_pt_path):
    encoder = vgg19_encoder().to(device)
    decoder = vgg19_decoder().to(device)

    encoder_pt = torch.load(encoder_pt_path)
    decoder_pt = torch.load(decoder_pt_path)

    encoder.load_state_dict(encoder_pt, strict=False)
    decoder.load_state_dict(decoder_pt)

    encoder = encoder.eval()
    decoder = decoder.eval()
    return encoder, decoder
