import os
import requests
from dotenv import load_dotenv

load_dotenv()

key_id = os.getenv("RAZORPAY_KEY_ID")
key_secret = os.getenv("RAZORPAY_KEY_SECRET")

print("Key ID:", key_id)
print("Secret loaded:", bool(key_secret))

response = requests.get(
    "https://api.razorpay.com/v1/orders",
    auth=(key_id, key_secret)
)

print("HTTP status:", response.status_code)
print("Response:")
print(response.text)