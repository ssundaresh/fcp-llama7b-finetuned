import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import login

# Login to Hugging Face Hub (replace with your actual token)
login("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")

# Define model and dataset names
model_name = "meta-llama/Llama-2-7b-hf"
dataset_name = "Biddls/Onion_News"
new_model = "llama-7b-ghibli"

# Initialize distributed training
dist.init_process_group(backend='nccl')
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# Load model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_cache=False,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# PEFT configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

# Apply PEFT to the model before wrapping with DDP
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Wrap the model with DDP
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

# Define the optimizer after wrapping with DDP
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Mixed precision scaler for efficient training
scaler = torch.cuda.amp.GradScaler()

# Load dataset
print("Loading dataset...")
dataset = load_dataset(dataset_name, split="train")
print("Dataset loaded successfully!")

# Training arguments
training_args = TrainingArguments(
    output_dir=new_model,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_torch",
    logging_steps=10,
    save_steps=100,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="none",
)

# Trainer
print("Initializing trainer...")
trainer = SFTTrainer(
    model=model.module,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    optimizers=(optimizer, None)  # Pass the optimizer to the trainer
)
print("Trainer initialized successfully!")

# Training loop
def train_model(trainer, scaler):
    model.train()
    dist.barrier()  # Ensure all processes are synchronized before training starts

    for epoch in range(training_args.num_train_epochs):
        total_train_loss = 0

        for step, batch in enumerate(trainer.get_train_dataloader()):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            # Backpropagation with mixed precision
            scaler.scale(loss).backward()

            # Update model parameters
            if (step + 1) % training_args.gradient_accumulation_steps == 0 or (step + 1) == len(trainer.get_train_dataloader()):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_train_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f"[Rank {local_rank}] Epoch {epoch+1}, Step {step+1}, Loss: {loss.item()}")

        print(f"[Rank {local_rank}] Epoch {epoch+1} completed. Average Loss: {total_train_loss / len(trainer.get_train_dataloader())}")

    # Save the model (only on the main process)
    if local_rank == 0:
        print("Saving the fine-tuned model...")
        model.module.save_pretrained(new_model)
        tokenizer.save_pretrained(new_model)
        print(f"Model saved at {new_model}")

# Run training
print("Starting training...")
train_model(trainer, scaler)
print("Training completed!")

# Clean up the process group
dist.destroy_process_group()