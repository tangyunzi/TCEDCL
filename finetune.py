import os
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import json
import random
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Define  event types
event_types = ["Phishing", "DataBreach", "Ransom", "DDosAttack", "Malware", "SupplyChain", "VulnerabilityImpact",
               "VulnerabilityDiscover", "VulnerabilityPatch","NA"]
# Load pre_trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('my-unsup-simcse-bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('my-unsup-simcse-bert-base-uncased',num_labels=len(event_types))

# Prepare data for fine-tuning
with open("data/train.jsonl", "r", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f]

max_length = 256  # set maximum length
train_dataset = []
for event in train_data:

    inputs = tokenizer.encode_plus(event["content"][0]["sentence"], add_special_tokens=True, padding='max_length',
                                   max_length=max_length)
    label_id = event_types.index(event["content"][0]["eventype"])
    train_dataset.append({'input_ids': inputs['input_ids'],
                     'attention_mask': inputs['attention_mask'],
                     'labels': label_id})

with open("data/dev.jsonl", "r", encoding="utf-8") as f:
    dev_data = [json.loads(line) for line in f]
eval_dataset = []
for event in dev_data:
    inputs = tokenizer.encode_plus(event["content"][0]["sentence"], add_special_tokens=True, padding='max_length',
                                   max_length=max_length)
    label_id = event_types.index(event["content"][0]["eventype"])
    eval_dataset.append({'input_ids': inputs['input_ids'],
                     'attention_mask': inputs['attention_mask'],
                     'labels': label_id})

# Define training parameters for fine-tuning
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_total_limit=5,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=lambda data: {'input_ids': torch.tensor([x['input_ids'] for x in data], dtype=torch.long),
                                'attention_mask': torch.tensor([x['attention_mask'] for x in data], dtype=torch.long),
                                'labels': torch.tensor([x['labels'] for x in data],dtype=torch.long)},
   tokenizer=tokenizer
)
trainer.train()
# Save fine-tuned model weights
trainer.save_model('saved_model')