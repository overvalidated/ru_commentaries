### !! REQUIRES COMPLETE REWRITING TO GET RID OF TRAINER API
### Usage of two separate tokenizers brings the requirement of separate data collators and different tokens inside of the model (probably).
### Using custom training loop will resolve this problems.
# %%
from typing import List

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


import pickle as pkl

import transformers

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# %%
output_dir = "./encoder_decoder_rugpt"
num_epochs = 2
batch_size = 128
micro_batch_size = 8
learning_rate = 5e-5
gradient_accumulation_steps = batch_size // micro_batch_size
# %%
tokenizer1 = transformers.AutoTokenizer.from_pretrained(
    "Salesforce/codegen-350M-nl", use_fast=True
)
tokenizer2 = transformers.GPT2Tokenizer.from_pretrained(
    "ai-forever/rugpt3medium_based_on_gpt2",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    use_fast=True,
)

# tokenizer1.add_special_tokens({'additional_special_tokens': ['<comment>']})
model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
    "Salesforce/codegen-350M-nl", "ai-forever/rugpt3medium_based_on_gpt2"
)

for params in model.encoder.parameters():
    params.requires_grad = False


# %%
def generate_and_tokenize_prompt_mydata(data_point):
    tokenized_full_prompt = tokenizer1(
        str(data_point["code"]),
        truncation=True,
        max_length=384,
        return_tensors=None,
    )
    tokenized_full_prompt["labels"] = tokenizer2(
        str(data_point["comment"]),
        truncation=True,
        max_length=256,
        return_tensors=None,
    ).input_ids
    # tokenized_user_prompt = tokenizer1(
    #     str(user_prompt) + "<comment>", truncation=True, max_length=512, padding='max_length', return_tensors=None,
    # )
    # user_prompt_len = len(tokenized_user_prompt)
    # tokenized_full_prompt["labels"] = [
    #     -100
    # ] * user_prompt_len + tokenized_full_prompt["labels"][
    #     user_prompt_len:
    # ]
    return tokenized_full_prompt


## loading data
with open("llama_comments_7b_final.pkl", "rb") as f:
    data_eng = pkl.load(f)
with open("llama_comments_7b_final_translated.pkl", "rb") as f:
    data_translated = pkl.load(f)
with open("llama_comments_13b_final_saiga_1.pkl", "rb") as f:
    data_translated2 = pkl.load(f)
with open("llama_comments_13b_final_saiga_2.pkl", "rb") as f:
    data_translated3 = pkl.load(f)
with open("llama_comments_13b_final_saiga.pkl", "rb") as f:
    data_translated4 = pkl.load(f)

data_translated = pd.DataFrame(
    {
        "code": list(data_eng.keys()),
        "comment": list(data_translated.values()),
        "lang": "ru",
    }
)
data_translated.comment = data_translated.comment.str.replace("<unk>", "")
data_translated.comment = data_translated.comment.str.replace("</s>", "")
data_translated.comment = data_translated.comment.str.strip()
data_translated = data_translated[data_translated.comment.map(len) > 40]
data_translated2 = pd.DataFrame(
    {
        "code": list(data_translated2.keys()),
        "comment": list(data_translated2.values()),
        "lang": "ru",
    }
)
data_translated2.comment = data_translated2.comment.str.replace("<unk>", "")
data_translated2.comment = data_translated2.comment.str.replace("</s>", "")
data_translated2.comment = data_translated2.comment.str.strip()
data_translated2 = data_translated2[data_translated2.comment.map(len) > 40]
data_translated3 = pd.DataFrame(
    {
        "code": list(data_translated3.keys()),
        "comment": list(data_translated3.values()),
        "lang": "ru",
    }
)
data_translated3.comment = data_translated3.comment.str.replace("<unk>", "")
data_translated3.comment = data_translated3.comment.str.replace("</s>", "")
data_translated3.comment = data_translated3.comment.str.strip()
data_translated3 = data_translated3[data_translated3.comment.map(len) > 40]
data_translated4 = pd.DataFrame(
    {
        "code": list(data_translated4.keys()),
        "comment": list(data_translated4.values()),
        "lang": "ru",
    }
)
data_translated4.comment = data_translated4.comment.str.replace("<unk>", "")
data_translated4.comment = data_translated4.comment.str.replace("</s>", "")
data_translated4.comment = data_translated4.comment.str.strip()
data_translated4 = data_translated4[data_translated4.comment.map(len) > 40]
print(data_translated4.head())
data_translated = pd.concat(
    [data_translated, data_translated2, data_translated3, data_translated4]
)
data = datasets.Dataset.from_pandas(data_translated).shuffle(seed=4112)

# Splitting train and test. Tokenization.
train_val = data.train_test_split(test_size=0.1, shuffle=True, seed=42)
print("orig", train_val["train"][:10])
train_data = (
    train_val["train"]
    .shuffle(seed=42)
    .map(
        generate_and_tokenize_prompt_mydata,
        remove_columns=train_val["train"].column_names,
        num_proc=12,
    )
)
val_data = (
    train_val["test"]
    .shuffle(seed=42)
    .map(
        generate_and_tokenize_prompt_mydata,
        remove_columns=train_val["test"].column_names,
        num_proc=12,
    )
)
# %%
tokenizer1.pad_token = tokenizer1.eos_token
model.config.decoder_start_token_id = 1
model.config.pad_token_id = 0
# %%

tok1_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer1, padding=True, pad_to_multiple_of=8
)
tok2_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer2, padding=True, pad_to_multiple_of=8
)


def data_collator(batch):
    collated1 = tok1_collator(batch)
    collated2 = tok2_collator(batch)
    if "decoder_input_ids" in collated2:
        collated1["decoder_input_ids"] = collated2["decoder_input_ids"]
    collated1["labels"] = collated2["labels"]
    return collated1


# %%
trainer_args = transformers.Seq2SeqTrainingArguments(
    per_device_train_batch_size=micro_batch_size,
    per_device_eval_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=100,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    bf16=True,
    tf32=True,
    optim="adamw_torch",
    weight_decay=0.001,
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=100,
    predict_with_generate=True,
    output_dir=output_dir,
    save_total_limit=1,
    lr_scheduler_type="cosine_with_restarts",
    ddp_find_unused_parameters=False,
    report_to=None,  # "wandb" if use_wandb else None,
    run_name=None,
)
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=trainer_args,
    data_collator=data_collator,
)


def compute_metrics(eval_preds):
    print(eval_preds.shape)
    metric = evaluate.load("sacrebleu")
    logits, labels = eval_preds
    print(logits.shape, labels.shape)
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


model = torch.compile(model)
trainer.train()  # resume_from_checkpoint=resume_from_checkpoint)

model.save_pretrained(output_dir)

# %%
sacrebleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

with torch.inference_mode():
    gts = []
    preds = []
    for batch in tqdm(train_val["test"].shuffle(seed=42)):
        full_prompt = batch["code"] + "<comment>"
        # tokenized_full_prompt = tokenize(full_prompt, tensors='pt', add_eos_token=False).to('cuda:0')
        tokenized_full_prompt = tokenizer1(
            full_prompt, truncation=True, max_length=512, return_tensors="pt"
        ).to("cuda:0")

        predict = tokenizer2.decode(
            model.generate(
                input_ids=tokenized_full_prompt["input_ids"],
                num_beams=1,
                max_new_tokens=256,
            )
            .cpu()
            .numpy()[0],
            skip_special_tokens=True,
        ).split("<comment>")[-1]
        preds.append(predict)
        gts.append([batch["comment"]])
    print("3b blue", sacrebleu.compute(predictions=preds, references=gts)["score"])
    print("3b rouge2", rouge.compute(predictions=preds, references=gts))


# %%
