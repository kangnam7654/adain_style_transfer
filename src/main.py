from pathlib import Path
import yaml
import argparse

import torch
from torchvision.utils import save_image

from utils.adain import adaptive_instance_normalization
from utils.load_transform import load_transform
from utils.image_loader import image_loader
from model.vgg19_encoder import vgg19_encoder
from model.vgg19_decoder import vgg19_decoder

ROOT_DIR = Path(__file__).parent

def style_transfer(encoder, decoder, content_input, style_input, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_feature = encoder(content_input)
    style_feature = encoder(style_input)
    feature = adaptive_instance_normalization(content_feature, style_feature)
    feature = feature * alpha + content_feature * (1 - alpha)
    return decoder(feature)

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.full_load(f)

    device = config['main']['device']

    # Image 불러오기
    content_path = config['main']['content_image']
    style_path = config['main']['style_image']
    transform = load_transform() # 이미지 변환 load
    content_input = image_loader(content_path, transform=transform, device=device) # [1, C, H, W] torch.tensor
    style_input = image_loader(style_path, transform=transform, device=device) # [1, C, H, W] torch.tensor


    # 모델 및 weight 로드
    encoder = vgg19_encoder().to(device)
    decoder = vgg19_decoder().to(device)

    encoder_pt_path = config['main']['encoder_pt_path']
    decoder_pt_path = config['main']['decoder_pt_path']

    encoder_pt = torch.load(encoder_pt_path)
    decoder_pt = torch.load(decoder_pt_path)

    encoder.load_state_dict(encoder_pt, strict=False)
    decoder.load_state_dict(decoder_pt)

    encoder = encoder.eval()
    decoder = decoder.eval()
    
    # Style Transfer
    with torch.no_grad():
        output = style_transfer(encoder=encoder,
                                decoder=decoder,
                                content_input=content_input,
                                style_input=style_input,
                                alpha=config['main']['alpha'])
    output = output.cpu()
    save_image(output, config['main']['save_path'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=ROOT_DIR.joinpath('config', 'config.yaml'), help='config의 경로입니다.')
    args = parser.parse_args()

    main(args=args)
