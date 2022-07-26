from pathlib import Path
import warnings
import argparse

import yaml
import torch
from torchvision.utils import save_image

from utils.load_transform import load_transform
from utils.image_loader import image_loader
from utils.load_codec import load_codec

from model.style_transfer import AdaIN_transfer

# 불필요한 경고 출력을 방지합니다.
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.full_load(f)

    device = config['main']['device']

    # image 불러오기
    content_path = config['main']['content_image']
    style_path = config['main']['style_image']
    transform = load_transform() # 이미지 변환 load
    content_input = image_loader(content_path, transform=transform, device=device) # [1, C, H, W] torch.tensor
    style_input = image_loader(style_path, transform=transform, device=device) # [1, C, H, W] torch.tensor

    # 모델 및 weight 로드
    encoder, decoder = load_codec(device, config)
    
    # Style Transfer
    with torch.no_grad():
        output = AdaIN_transfer(encoder=encoder,
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
