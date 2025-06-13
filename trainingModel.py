from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import torch

# Load your dataset
dataset = load_dataset("json", data_files="finance_questions_formatted.json")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Tokenize data
def preprocess(example):
    prompt = f"Explain the following finance concept:\n\n{example['question']}"
    return tokenizer(prompt, text_target=example["answer"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess, batched=True)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./flan-finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs"
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"],  # For real use, split into train/test
    tokenizer=tokenizer
)

# Train and save
trainer.train()
trainer.save_model("./flan-finetuned")
tokenizer.save_pretrained("./flan-finetuned")
