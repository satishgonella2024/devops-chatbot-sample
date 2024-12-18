from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

model_name = "microsoft/DialoGPT-medium"  # Try a larger model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class ChatRequest(BaseModel):
    input_text: str

@app.post("/chat")
def chat(request: ChatRequest):
    prompt = (
        "You are a friendly and helpful assistant. The user is greeting you and asking how you are.\n"
        "Please reply in a friendly, coherent manner.\n\n"
        f"User: {request.input_text}\n"
        "Assistant:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            temperature=0.5,  # Lower temperature for more deterministic output
            top_k=40,
            top_p=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Try to parse out just the assistant part if needed
    if "Assistant:" in response_text:
        response_text = response_text.split("Assistant:")[-1].strip()
    
    return {"response": response_text}
