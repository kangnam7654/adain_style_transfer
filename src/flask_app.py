from flask import Flask


app = Flask(__name__)

@app.route("/")
@app.route("/home")
def test():
    return "<h1>헬로 Flask!!!</h1>"

class StyleTransfer:
    def __init__(self) -> None:
        pass

if __name__ == "__main__":
    app.run(host='0.0.0.0',
            debug=True)