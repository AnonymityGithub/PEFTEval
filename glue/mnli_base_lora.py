import torch
import logging
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed



batch_size = 32
model_name_or_path = "./Roberta_base"
task = "mnli"
model_save_path = "./result_with_loss/checkpoint/mnli_base_lora/"

logging.basicConfig(filename='./result_with_loss/loss/mnli_base_lora.log', level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
device = torch.device("cuda:7")

peft_type = PeftType.LORA
num_epochs = 60

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
lr = 3e-4

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("./GLUE/glue.py", task)

metric = evaluate.load("glue", task)


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=510)
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

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=3)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
optimizer = AdamW(params=model.parameters(), lr=lr)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model.to(device)
acc_list = []
acc_m_list = []
adversarial_list = []
adversarial_m_list = []
for epoch in range(num_epochs):
    model.train()
    batch_idx = 0
    for step, batch in enumerate(train_dataloader):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        batch_idx = batch_idx + 1
    model.save_pretrained(model_save_path + str(epoch))

    model.eval()
    for step, batch in enumerate(eval_dataloader):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    eval_metric = metric.compute()
    acc_list.append(eval_metric)
    print(f"epoch {epoch}:", eval_metric)
    for i in range(len(acc_list)):
        print("epoch", i, ": ", acc_list[i])

    for step, batch in enumerate(eval_m_dataloader):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    eval_metric = metric.compute()
    acc_m_list.append(eval_metric)
    print(f"epoch {epoch}:", eval_metric)
    for i in range(len(acc_list)):
        print("epoch", i, ": ", acc_m_list[i])

    # adversarial validation
    for step, batch in enumerate(test_dataloader):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    adversarial_list.append(eval_metric)
    print(f"epoch {epoch}:", eval_metric)
    for i in range(len(adversarial_list)):
        print("epoch", i, ": ", adversarial_list[i])


    for step, batch in enumerate(test_m_dataloader):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    adversarial_m_list.append(eval_metric)
    print(f"epoch {epoch}:", eval_metric)
    for i in range(len(adversarial_m_list)):
        print("epoch", i, ": ", adversarial_m_list[i])