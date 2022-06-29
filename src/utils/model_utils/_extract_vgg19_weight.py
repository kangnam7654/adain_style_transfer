import timm
import torch


def extract_original_weights(save_path):
    vgg = timm.create_model('vgg19', pretrained=True)
    torch.save(vgg.state_dict(), save_path)

if __name__ == '__main__':
    import os
    from pathlib import Path

    WEIGHT_DIR = os.path.join(Path(__file__).parent, 'weights')
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    extract_original_weights(os.path.join(WEIGHT_DIR, 'vgg19_original.pt'))
