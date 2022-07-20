from pathlib import Path
import warnings
import argparse

import cv2
import yaml
import torch

from utils.adain import adaptive_instance_normalization
from utils.load_transform import load_transform
from utils.image_loader import image_loader
from model.vgg19_encoder import vgg19_encoder
from model.vgg19_decoder import vgg19_decoder


# 불필요한 경고 출력을 방지합니다.
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent


def load_codec(device, config):
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
    return encoder, decoder

def style_transfer(encoder, decoder, content_input, style_input, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_feature = encoder(content_input)
    style_feature = encoder(style_input)
    feature = adaptive_instance_normalization(content_feature, style_feature)
    feature = feature * alpha + content_feature * (1 - alpha) # Alpha가 1에 가까울수록 스타일이 진해짐
    return decoder(feature)

def main():
    with open(ROOT_DIR.joinpath('config', 'config.yaml'), 'r') as f:
        config = yaml.full_load(f)

    device = config['main']['device']

    # image 불러오기
    content_path = 'D:\project\style_transfer\AdaIN_style_transfer\src\data\sample.mp4' # video
    style_path = config['main']['style_image']
    transform = load_transform() # 이미지 변환 load
    style_input = image_loader(style_path, transform=transform, device=device) # [1, C, H, W] torch.tensor

    out = cv2.VideoWriter(filename='test2.mp4', fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=25, frameSize=(512,512))
    cap = cv2.VideoCapture(content_path)
    assert(cap.isOpened())
    encoder, decoder = load_codec(device, config)

    while True:
        ret, frame = cap.read()
        if ret:
            content = image_loader(frame, transform, device)
            st = style_transfer(encoder, decoder, content_input=content, style_input=style_input, alpha=0.5)
            st = st.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            out.write(st)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    # Style Transfer
    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default=ROOT_DIR.joinpath('config', 'config.yaml'), help='config의 경로입니다.')
    # args = parser.parse_args()

    main()
