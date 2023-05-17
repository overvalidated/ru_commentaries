### !! REQUIRES COMPLETE REWRITING TO GET RID OF TRAINER API
### Usage of two separate tokenizers brings the requirement of separate data collators and different tokens inside of the model (probably).
### Using custom training loop will resolve this problems.

from typing import List

import datasets
import fire
import pandas as pd
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import transformers
import pickle as pkl


def train(
    output_dir: str = "./encoder_decoder",
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 5,
    learning_rate: float = 2e-4,
    val_set_size: int = 2000,
):
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
    # os.environ["RANK"] = "0"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    batch_size = 256
    micro_batch_size = 8
    learning_rate = 3e-5
    gradient_accumulation_steps = batch_size // micro_batch_size

    tokenizer1 = transformers.AutoTokenizer.from_pretrained(
        "Salesforce/codegen-350M-multi", use_fast=True
    )
    tokenizer2 = transformers.GPT2Tokenizer.from_pretrained(
        "ai-forever/rugpt3medium_based_on_gpt2",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        use_fast=True,
    )
    # tokenizer1.add_special_tokens({'additional_special_tokens': ['<ru>', '<en>']})

    model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
        "Salesforce/codegen-350M-multi", "ai-forever/rugpt3medium_based_on_gpt2"
    )
    # model.decoder.resize_token_embeddings(
    #     len(tokenizer1)
    # )
    for params in model.encoder.parameters():
        params.requires_grad = False

    tokenizer1.padding_side = "left"  # Allow batched inference
    # tokenizer2.padding_side = "left"  # Allow batched inference

    def generate_and_tokenize_prompt_mydata(data_point):
        user_prompt = data_point["code"]
        tokenized_full_prompt = tokenizer1(
            user_prompt, truncation=True, max_length=768, return_tensors=None
        )
        tokenized_full_prompt["labels"] = tokenizer2(
            data_point["comment"], truncation=True, max_length=128, return_tensors=None
        )["input_ids"]
        return tokenized_full_prompt

    with open("llama_comments_translated.pkl", "rb") as f:
        data_translated = pkl.load(f)
    data_translated = pd.DataFrame(
        {
            "code": list(data_translated.keys()),
            "comment": list(data_translated.values()),
            "lang": "ru",
        }
    )
    print(data_translated.head())

    with open("llama_comments.pkl", "rb") as f:
        data = pkl.load(f)
    data = pd.DataFrame(
        {"code": list(data.keys()), "comment": list(data.values()), "lang": "en"}
    )
    # data = pd.concat([data, data_translated])

    data = datasets.Dataset.from_pandas(data_translated).shuffle(seed=4112)

    # tokenizer1.pad_token_id = 50256#tokenizer1.eos_token_id
    # tokenizer2.pad_token_id = 50256#tokenizer2.eos_token_id

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
    print(
        "pad_tokens", tokenizer1.pad_token, tokenizer2.pad_token, tokenizer2.bos_token
    )
    model.config.decoder_start_token_id = (
        tokenizer2.bos_token_id
    )  # tokenizer2.bos_token_id
    model.config.pad_token_id = tokenizer1.bos_token_id
    tokenizer1.pad_token_id = tokenizer1.bos_token_id

    trainer_args = transformers.Seq2SeqTrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # gradient_checkpointing=True,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        # bf16=True,
        tf32=True,
        # weight_decay=0.001,
        logging_steps=10,
        # max_steps=300,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=100 if val_set_size > 0 else None,
        save_steps=100,
        dataloader_num_workers=0,
        output_dir=output_dir,
        save_total_limit=3,
        ddp_find_unused_parameters=False,
        report_to=None,  # "wandb" if use_wandb else None,
        run_name=None,
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=trainer_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer1, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # compute_metrics=compute_metrics
    )

    model = torch.compile(model)
    trainer.train()  # resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
