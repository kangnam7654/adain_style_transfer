import base64
from pathlib import Path

import torch
from torchvision.utils import save_image
from flask import Flask, request, render_template

from model.style_transfer import AdaIN_transfer
from utils.image_loader import image_loader
from utils.load_transform import load_transform
from utils.load_codec import load_codec


ROOT_DIR = Path(__file__).parent
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder_pt_path = ROOT_DIR.joinpath('model', 'weights', 'vgg19_encoder.pt')
decoder_pt_path = ROOT_DIR.joinpath('model', 'weights', 'vgg19_decoder.pt')
save_path = ROOT_DIR.joinpath('data')
encoder, decoder = load_codec(encoder_pt_path=encoder_pt_path, decoder_pt_path=decoder_pt_path, device=device)
transform = load_transform()



app = Flask(__name__)

@app.route('/')
@app.route('/upload', methods=['GET'])
def upload_page():
    if request.method == 'GET':
        return render_template("1_upload.html")

@app.route("/inference", methods=['POST'])
def inference():
    if request.method == 'POST':
        content_input = request.files['content_input']
        style_input = request.files['style_input']

        con_save_path = save_path.joinpath(content_input.filename).absolute()
        stl_save_path = save_path.joinpath(style_input.filename).absolute()
        result_file_name = f'{content_input.filename.split(".")[0]}_{style_input.filename.split(".")[0]}.jpg'
        result_save_path = ROOT_DIR.joinpath('result', result_file_name)

        content_input.save(con_save_path)
        style_input.save(stl_save_path)

        content = image_loader(con_save_path, transform=transform, device=device)
        style = image_loader(stl_save_path, transform=transform, device=device)
        output = AdaIN_transfer(encoder, decoder, content, style, 0.8)
        save_image(output, result_save_path)
        with open(result_save_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read())
        return image_b64

if __name__ == "__main__":
    app.run(host='localhost', debug=True)
