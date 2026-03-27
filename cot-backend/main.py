import os
import json
import re
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Suppress the deprecation warning for google.generativeai
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    print(f"SUCCESS: API Key loaded (starts with: {api_key[:8]}...)")
else:
    print("ERROR: GOOGLE_API_KEY not found in .env file!")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('models/gemini-1.5-flash')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/chat")
async def generate_cot(request: ChatRequest):
    try:
        system_instr = "Respond ONLY with JSON. Schema: {\"thoughts\": [{\"step\": \"Analysis\", \"content\": \"...\"}], \"final_answer\": \"...\"}"
        full_prompt = f"{system_instr}\n\nUser Question: {request.prompt}"
        
        response = model.generate_content(full_prompt)
        print(f"AI Response received for: {request.prompt[:20]}...")

        clean_text = re.sub(r'```json|```', '', response.text).strip()
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        
        if not match:
            return {"thoughts": [], "final_answer": response.text}
            
        return json.loads(match.group(0))
    except Exception as e:
        print(f"CRASH DURING API CALL: {str(e)}")
        return {
            "thoughts": [{"step": "System Error", "content": str(e)}],
            "final_answer": "Check your terminal for the error log."
        }

class ReasonRequest(BaseModel):
    query: str

@app.post("/api/reason")
async def generate_reason(request: ReasonRequest):
    try:
        response = model.generate_content(request.query)
        print(f"AI Reason Response received for: {request.query[:20]}...")
        return {"answer": response.text}
    except Exception as e:
        print(f"CRASH DURING REASON API CALL: {str(e)}")
        return {"answer": f"Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)