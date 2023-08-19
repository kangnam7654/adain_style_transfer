from pathlib import Path
import warnings

import cv2
import yaml

from model.style_transfer import AdaIN_transfer
from utils.load_transform import load_transform
from utils.image_loader import image_loader, cv2pil
from utils.tensor2cv import tensor2cv
from utils.load_codec import load_codec


# 불필요한 경고 출력을 방지합니다.
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent

def main():
    with open(ROOT_DIR.joinpath('config', 'config.yaml'), 'r') as f:
        config = yaml.full_load(f)

    device = config['main']['device']

    # image 불러오기
    content_path = config['main']['content'] # video
    style_path = config['main']['style']
    transform = load_transform() # 이미지 변환 load
    
    cap = cv2.VideoCapture(content_path) # Content Video
    style_input = image_loader(style_path, transform=transform, device=device) # [1, C, H, W] torch.tensor

    fourcc = cv2.VideoWriter_fourcc(*config['main']['fourcc'])
    out = cv2.VideoWriter(fourcc=fourcc, **config['main']['video_writer']) # Video 저장 객체  
    
    assert(cap.isOpened())
    encoder, decoder = load_codec(device=device, **config['main']['load_codec'])

    while True:
        ret, frame = cap.read()
        if ret:
            content = cv2pil(frame, transform, device)
            st = AdaIN_transfer(encoder, decoder, content_input=content, style_input=style_input, alpha=0.8)
            out.write(tensor2cv(st))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()

   
if __name__ == '__main__':
    main()
