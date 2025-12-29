import base64, json, time
from cryptography.fernet import Fernet, InvalidToken
import os
from dotenv import load_dotenv

load_dotenv()

SECRET = os.getenv("DOOR_SECRET_KEY")

fernet = Fernet(base64.urlsafe_b64encode(
    base64.b64decode(SECRET)
))

def decrypt_request(ciphertext: str):
    try:
        decrypted = fernet.decrypt(ciphertext.encode(), ttl=10)
        payload = json.loads(decrypted)

        # Anti-replay
        now = int(time.time())
        if abs(now - payload["timestamp"]) > 10:
            raise ValueError("Expired request")

        return payload

    except (InvalidToken, ValueError) as e:
        return None
