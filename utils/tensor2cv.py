import torch

def tensor2cv(model_output):
    cv_image = model_output.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return cv_image
