from flask import Flask, request

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    print("\nWEBHOOK RECEIVED:")
    print(data)
    return "OK", 200

app.run(port=5000)