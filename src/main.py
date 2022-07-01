import torch
from torchvision.utils import save_image

from utils.adain import adaptive_instance_normalization
from utils.load_transform import load_transform
from utils.image_loader import image_loader
from model.vgg19_encoder import vgg19_encoder
from model.vgg19_decoder import vgg19_decoder


def style_transfer(encoder, decoder, content_input, style_input, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_feature = encoder(content_input)
    style_feature = encoder(style_input)
    feature = adaptive_instance_normalization(content_feature, style_feature)
    feature = feature * alpha + content_feature * (1 - alpha)
    return decoder(feature)

def main():
    # TODO: device 수정
    device = 'cuda'
    ###

    content_path = 'G:\project\style_transfer\AdaIN_style_transfer\src\data\\piddle.jpg'
    style_path = 'G:\project\style_transfer\AdaIN_style_transfer\src\data\\picasso_mo.jpg'
    transform = load_transform()
    content_input = image_loader(content_path, transform=transform, device=device)
    style_input = image_loader(style_path, transform=transform, device=device)

    encoder = vgg19_encoder().to(device)
    decoder = vgg19_decoder().to(device)

    # weight 로드
    encoder_pt_path = 'G:\project\style_transfer\AdaIN_style_transfer\src\model\weights\\vgg19_encoder.pt'
    decoder_pt_path = 'G:\project\style_transfer\AdaIN_style_transfer\src\model\weights\\vgg19_decoder.pt'

    encoder_pt = torch.load(encoder_pt_path)
    decoder_pt = torch.load(decoder_pt_path)

    encoder.load_state_dict(encoder_pt, strict=False)
    decoder.load_state_dict(decoder_pt)


    with torch.no_grad():
        output = style_transfer(encoder=encoder,
                                decoder=decoder,
                                content_input=content_input,
                                style_input=style_input,
                                alpha=0.5)
    output = output.cpu()
    save_image(output, 'output.png')

if __name__ == '__main__':
    main()
