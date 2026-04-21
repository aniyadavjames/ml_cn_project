import os
import sys
from transformers import (
    BertForMaskedLM, BertTokenizer, 
    DataCollatorForLanguageModeling, 
    Trainer, TrainingArguments,
    TrainerCallback
)
from datasets import load_dataset

# 1. Force Offline Mode for IITD Compute Nodes
os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['HF_DATASETS_OFFLINE'] = "1"

# 2. Local Paths
model_path = "/scratch/cse/phd/csz258233/col7560/et-bert/base_model"
train_file = "/scratch/cse/phd/csz258233/col7560/et-bert/corpora/encrypted_burst_final.txt"
output_dir = "/scratch/cse/phd/csz258233/col7560/et-bert/retrained_model"

# Custom Status Printer
class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero: # Only print from the main GPU
            print(f"Step: {state.global_step} | Loss: {logs.get('loss', 'N/A')} | LR: {logs.get('learning_rate', 'N/A')}")
            sys.stdout.flush()

# 3. Load Model & Tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)

# 4. Load Dataset (Utilizing 400GB RAM)
# keep_in_memory=True ensures the 56GB corpus stays in RAM for both GPUs
dataset = load_dataset("text", data_files=train_file, split="train", keep_in_memory=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    num_proc=24, 
    remove_columns=["text"],
    keep_in_memory=True
)

# 5. MLM Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# 6. Training Arguments for Multi-GPU
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16, # Total batch size will be 16 * 2 GPUs = 32
    gradient_accumulation_steps=4,  # Effective batch size = 32 * 4 = 128
    save_steps=5000,
    save_total_limit=2,
    fp16=True,                      
    logging_steps=100,
    report_to="none",
    ddp_find_unused_parameters=False, # Optimization for BERT
    dataloader_num_workers=4          # Parallel data loading to feed the GPUs
)

# 7. Start Training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    callbacks=[PrinterCallback()]
)

trainer.train()
trainer.save_model(output_dir)
