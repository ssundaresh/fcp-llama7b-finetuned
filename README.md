# 🦙 LLaMA 2 Fine-tuning and API Deployment

This project fine-tunes the LLaMA 2 (7B) model using LoRA (Low-Rank Adaptation) and serves the fine-tuned model via a FastAPI endpoint. It is designed to generate text based on user prompts, making it suitable for various NLP tasks.


🚀 Features

    * Fine-tuning with LoRA: Efficient fine-tuning of the LLaMA 2 model using the LoRA technique.
    * Distributed Training: Leverages mixed precision and distributed training for faster model optimization.
    * API Endpoint: A FastAPI server to generate text using the fine-tuned model.

### 📁 Project Structure

    ├── app.py      # FastAPI server for text generation

    ├── eval_finetunedModel.py   # Script to evaluate the fine-tuned model

    ├── llama-7b-onionnews    #Folder with finetuned model used by app.py

    ├── llama7b-onionnews-finetuned.py  # Training script for fine-tuning LLaMA 2 using LoRA

### 🛠️ Requirements

  Install dependencies by running:
  
    pip3 install -r requirements.txt

### ⚙️ Setup & Usage

  1. Clone the Repository
      ```
      git clone https://github.com/ssundaresh/llama-finetuning.git
      cd llama-finetuning
  2. Fine-tune the Model
     If using 4 GPUs. Adjust according to your environment.
     ```
       torchrun --nproc_per_node=4 finetunedmodel/llama7b-ghibli-finetuned.py

  3. Start the API Server
     On port 8000
     ```
       uvicorn app:app --host 0.0.0.0 --port 8000 --reload
  4. Open the port
     ```
       sudo foundrypf 8000
  5. Generate text via the API
     ```
         curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" \
         -d '{"prompt": "Hello AI", "max_length": 50, "num_return_sequences": 1}'



