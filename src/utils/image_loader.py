from PIL import Image
import torch

def image_loader(image_name, transform, device):
    image = Image.open(image_name)
    image = transform(image).unsqueeze(0)  # 네트워크의 입력 차원에 맞추기 위해 필요한 가짜 배치 차원
    return image.to(device, torch.float)
