import torch
# import evaluate
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed


batch_size = 32
model_name_or_path = "./Roberta_large"
task = "mnli"

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


datasets = load_dataset("./GLUE/glue.py", task)
# print(datasets)
# metric = evaluate.load("glue", task)


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=None)
    return outputs


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "premise", "hypothesis"],
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["validation_matched"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

eval_m_dataloader = DataLoader(
    tokenized_datasets["validation_mismatched"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

test_datasets = load_dataset("json", data_files="./dev/" + task + ".json")
test_tokenized_datasets = test_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "premise", "hypothesis"],
)
test_tokenized_datasets = test_tokenized_datasets.rename_column("label", "labels")
test_dataloader = DataLoader(
    test_tokenized_datasets["train"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)
#  test mismatch
test_m_datasets = load_dataset("json", data_files="./dev/" + task + "_mm.json")
test_m_tokenized_datasets = test_m_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "premise", "hypothesis"],
)
test_m_tokenized_datasets = test_m_tokenized_datasets.rename_column("label", "labels")
test_m_dataloader = DataLoader(
    test_m_tokenized_datasets["train"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)


model = RobertaForSequenceClassification.from_pretrained(model_name_or_path, num_labels=3)

lr = 2e-5
num_epochs = 30


criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr)

total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

device = torch.device("cuda:4")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
test_list = []
test_m_list = []
val_list = []
val_m_list = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    model.save_pretrained("./mnli_large_ft/" + str(epoch))

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(eval_dataloader)
    val_accuracy = (val_correct / val_total) * 100
    val_list.append(val_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Training Loss: {avg_train_loss:.4f}')
    print(f'Validation Loss: {avg_val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

    val_m_loss = 0
    val_m_correct = 0
    val_m_total = 0

    with torch.no_grad():
        for batch in eval_m_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_m_loss += loss.item()

            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            val_m_total += labels.size(0)
            val_m_correct += (predicted == labels).sum().item()

    avg_val_m_loss = val_m_loss / len(eval_m_dataloader)
    val_m_accuracy = (val_m_correct / val_m_total) * 100
    val_m_list.append(val_m_accuracy)

    print(f'Validation mismatch Loss: {avg_val_m_loss:.4f}')
    print(f'Validation mismatch Accuracy: {val_m_accuracy:.2f}%')

    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            test_loss += loss.item()

            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_dataloader)
    test_accuracy = (test_correct / test_total) * 100
    test_list.append(test_accuracy)
    test_m_loss = 0
    test_m_correct = 0
    test_m_total = 0
    with torch.no_grad():
        for batch in test_m_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            test_loss += loss.item()

            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            test_m_total += labels.size(0)
            test_m_correct += (predicted == labels).sum().item()

    avg_test_m_loss = test_m_loss / len(test_m_dataloader)
    test_m_accuracy = (test_m_correct / test_m_total) * 100
    test_m_list.append(test_m_accuracy)
    print(f'Validation mismatch Loss: {avg_test_m_loss:.4f}')
    print(f'Validation mismatch Accuracy: {test_m_accuracy:.2f}%')

    for i in range(len(test_list)):
        print("test epoch ", i, ": ", val_list[i])
        print("test mismatch epoch ", i, ": ", val_m_list[i])
        print("adversarial epoch ", i, ": ", test_list[i])
        print("adversarial mismatch epoch ", i, ": ", test_m_list[i])
