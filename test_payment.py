import os
import razorpay
from dotenv import load_dotenv

load_dotenv()

key_id = os.getenv("RAZORPAY_KEY_ID")
key_secret = os.getenv("RAZORPAY_KEY_SECRET")

client = razorpay.Client(
    auth=(key_id, key_secret)
)

order_data = {
    "amount": 50000,  # ₹500
    "currency": "INR",
    "receipt": "fraud_research_001",
    "notes": {
        "experiment": "fraud_detection",
        "user_id": "test_user_001"
    }
}

order = client.order.create(data=order_data)

print("\nORDER CREATED")
print(order)