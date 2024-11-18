from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Define paths
base_model_name = "meta-llama/Llama-2-7b-hf"  # Base model
adapter_path = "./llama-7b-onionnews"  # Directory where your adapter is saved

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16).to(device)

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the fine-tuned adapter weights
print("Loading fine-tuned LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.to(device)
model.eval()
print("Model and adapter loaded successfully!")

# Initialize FastAPI
app = FastAPI()

class TextRequest(BaseModel):
    prompt: str
    max_length: int = 100
    num_return_sequences: int = 1

@app.post("/generate")
async def generate_text(request: TextRequest):
    try:
        # Tokenize the input prompt
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_length=request.max_length,
                num_return_sequences=request.num_return_sequences,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        # Decode and return the generated text
        generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in output]
        return {"generated_texts": generated_texts}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Llama-7B OnionNews Text Generation API!"}
