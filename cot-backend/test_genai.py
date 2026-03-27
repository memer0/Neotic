from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

for m in ['gemini-2.0-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-pro']:
    try:
        response = client.models.generate_content(
            model=m,
            contents='test'
        )
        print(f"SUCCESS {m}:", response.text[:20])
    except Exception as e:
        print(f"ERROR {m}:", repr(e))
