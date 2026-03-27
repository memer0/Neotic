import asyncio
import websockets
import json

async def run_demo():
    uri = "ws://localhost:8000/ws/chat"
    try:
        async with websockets.connect(uri) as websocket:
            # ask a question about the sample file
            question = "What is the mission brief of Project Noetic and what is its latest secret feature?"
            print(f"> Sending question: {question}\n")
            
            await websocket.send(json.dumps({"question": question}))

            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                step = data.get("step")
                content = data.get("content")
                
                if step == "Action":
                    print(f"THOUGHT: {content}")
                elif step == "Observation":
                    print(f"RETRIEVAL: {content[:100]}...")
                elif step == "Thought":
                    # For token streaming
                    print(content, end="", flush=True)
                elif step == "Final Answer":
                    print(f"\n\nANSWER: {content}")
                    break
    except Exception as e:
        print(f"Error connecting to backend: {e}")
        print("Make sure your main.py is running on port 8000.")

if __name__ == "__main__":
    import sys
    if "websockets" not in sys.modules:
        print("Please install websockets: pip install websockets")
    asyncio.run(run_demo())
