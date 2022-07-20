import numpy as np
import torch
from PIL import Image

def image_loader(image, transform, device):
    if type(image) == str:
        image = Image.open(image)
    elif type(image) == np.ndarray:
        image = Image.fromarray(image)
    else:
        raise Exception(f'wrong input "{image}"; \n path or cv2 image(numpy) needed')
    image = transform(image).unsqueeze(0)  # 네트워크의 입력 차원에 맞추기 위해 필요한 가짜 배치 차원
    return image.to(device, torch.float)
