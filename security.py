import base64, json, time
from cryptography.fernet import Fernet, InvalidToken
import os
from dotenv import load_dotenv

load_dotenv()

SECRET = os.getenv("DOOR_SECRET_KEY")
print("Loaded SECRET:", SECRET)

fernet = Fernet(SECRET.encode())

def decrypt_request(ciphertext: str):
    try:
        decrypted = fernet.decrypt(ciphertext.encode(), ttl=10)
        payload = json.loads(decrypted)
        print("Decrypted payload:", payload)
        # Anti-replay
        now = int(time.time())
        if abs(now - payload["timestamp"]) > 10:
            raise ValueError("Expired request")

        return payload

    except (InvalidToken, ValueError) as e:
        return None
