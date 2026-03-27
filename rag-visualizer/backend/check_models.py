import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: No GOOGLE_API_KEY found in .env")
else:
    client = genai.Client(api_key=api_key)
    print("Listing available models via Google GenAI SDK...")
    try:
        models = client.models.list()
        for m in models:
            print(f" - {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
