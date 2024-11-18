from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# Define the model and dataset names
model_name = "meta-llama/Llama-2-7b-hf"
dataset_name = "Biddls/Onion_News"
new_model = "./llama-7b-onionnews"

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained(new_model)
tokenizer = AutoTokenizer.from_pretrained(new_model)

# Load the dataset
dataset = load_dataset(dataset_name, split="train")

# Split the dataset into train and test
train_test_dataset = dataset.train_test_split(test_size=0.1, seed=42)  # 10% for test

# Access the train and test splits
train_dataset = train_test_dataset["train"]
test_dataset = train_test_dataset["test"]

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Training arguments (adjust as needed)
training_args = TrainingArguments(
    output_dir="./test_results",
    per_device_eval_batch_size=1,  # Adjust batch size as needed
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
)

# Evaluate the model
print("Evaluating the model...")
results = trainer.evaluate(eval_dataset=tokenized_test_dataset)  # Evaluate on tokenized test set
print(results)