from pathlib import Path
import gradio as gr
from utils.load_codec import load_codec
from model.style_transfer import AdaIN_transfer
from torchvision import transforms

encoder, decoder = load_codec(
    encoder_pt_path=Path(__file__).parent.joinpath(
        "model", "weights", "vgg19_encoder.pt"
    ),
    decoder_pt_path=Path(__file__).parent.joinpath(
        "model", "weights", "vgg19_decoder.pt"
    ),
    device="cuda",
)

transform = transforms.Compose([
    # transforms.Resize(512, 512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def greet(name):
    return "hello" + name + "!"

def preprocess(image):
    return transform(image)

# TODO 변수에 alpha 추가
# TODO 이미지 전처리 적용
def inference(content_image, style_image, alpha=1.0):
    styled = AdaIN_transfer(
        encoder=encoder,
        decoder=decoder,
        content_input=content_image,
        style_input=style_image,
        alpha=alpha,
    )
    return styled


def run():
    demo = gr.Interface(fn=inference, inputs=[gr.Image(), gr.Image()], outputs="image")
    demo.launch()


if __name__ == "__main__":
    run()
