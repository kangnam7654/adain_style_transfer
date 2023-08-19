import io
import torch
from PIL import Image

def image_loader(image, transform, device):
    image = Image.open(image)
    image = transform(image).unsqueeze(0)  # 네트워크의 입력 차원에 맞추기 위해 필요한 가짜 배치 차원
    return image.to(device, torch.float)


def cv2pil(cv2_image, transform, device):
    image = Image.fromarray(cv2_image)
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)


def io_image_open(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)) # PIL Image
    return image
