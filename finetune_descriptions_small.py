import datasets
import fire
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import transformers
import pickle as pkl
from huggingface_hub import hf_hub_download
from peft import get_peft_model_state_dict, PeftModel, LoraConfig, get_peft_model

from utils.prompter import Prompter

"""
batch_size = 256
micro_batch_size = 1
learning_rate = 1e-4
cutoff_len = 1024
"""

output_dir: str = "./lora-alpaca-small"
num_epochs: int = 2
val_set_size: int = 2000
train_on_inputs: bool = True  # if False, masks out inputs in loss
prompt_template_name: str = "openassistant"  # The prompt template to use, will default to openassistant.
batch_size = 512
micro_batch_size = 1
learning_rate = 7e-5
cutoff_len = 384+256
gradient_accumulation_steps = batch_size // micro_batch_size

prompter = Prompter(prompt_template_name)

tokenizer = transformers.LlamaTokenizer.from_pretrained(
    "openlm-research/open_llama_3b_350bt_preview"
)

config = transformers.LlamaConfig(**{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 5280,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 20,
  "pad_token_id": 0,
  "rms_norm_eps": 1e-06,
  "tie_word_embeddings": False,
  "torch_dtype": torch.float16,
  "use_cache": True,
  "vocab_size": 32000
})

model = transformers.LlamaForCausalLM(
    config
)
# model_emb = transformers.LlamaForCausalLM.from_pretrained(
#     "decapoda-research/llama-7b-hf", torch_dtype=torch.float16
# )  # Load Base Model

# model.set_input_embeddings(model_emb.get_input_embeddings())
# del model_emb
# model.resize_token_embeddings(
#     len(tokenizer)
# )  # This model repo also contains several embeddings for special tokens that need to be loaded.

tokenizer.pad_token_id = 0
tokenizer.bad_token_id = 1
tokenizer.eos_token_id = 2
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.bos_token_id = tokenizer.bos_token_id
# model.config.pad_token_id = tokenizer.pad_token_id

tokenizer.padding_side = "left"  # Allow batched inference

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt, truncation=True, max_length=cutoff_len, return_tensors=None
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt_mydata(data_point):
    # full_prompt = prompter.generate_prompt(
    #     "Explain this code in "
    #     + ("Russian." if data_point["lang"] == "ru" else "English.")
    #     + "Describe its purpose.",
    #     data_point["code"],
    #     data_point["comment"],
    # )
    full_prompt = data_point['code'] + "<comment>" + data_point['comment']
    tokenized_full_prompt = tokenize(full_prompt)
    # if not train_on_inputs:
    # user_prompt = prompter.generate_prompt(
    #     "Explain this code in "
    #     + ("Russian." if data_point["lang"] == "ru" else "English.")
    #     + "Describe its purpose.",
    #     data_point["code"]
    # )
    # user_prompt = data_point['code'] + "<comment>"
    # tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
    # user_prompt_len = len(tokenized_user_prompt["input_ids"])

    # tokenized_full_prompt["labels"] = [
    #     -100
    # ] * user_prompt_len + tokenized_full_prompt["labels"][
    #     user_prompt_len:
    # ]  # could be sped up, probably
    return tokenized_full_prompt

# model = PeftModel.from_pretrained(
#     model,
#     "jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b",
#     torch_dtype=torch.float16,
#     is_trainable=True,
# )
# model.print_trainable_parameters()

# model.eos_token_id = tokenizer.eos_token_id
# filename = hf_hub_download(
#     "jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b", "extra_embeddings.pt"
# )
# embed_weights = torch.load(
#     filename, map_location="cpu"
# )  # Load embeddings for special tokens
# model.base_model.model.model.embed_tokens.weight[32000:, :] = embed_weights.to(
#     model.base_model.model.model.embed_tokens.weight.dtype
# )  # Add special token embeddings
# for param in model.base_model.model.model.embed_tokens.parameters():
#     param.require_grad = True

model = model.to("cuda")
model.enable_input_require_grads()
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
data_translated.comment = data_translated.comment.str.replace('<unk>', '')
data_translated.comment = data_translated.comment.str.replace('</s>', '')
data_translated.comment = data_translated.comment.str.strip()
data_translated = data_translated[data_translated.comment.map(len) > 40]
data_translated2 = pd.DataFrame(
    {
        "code": list(data_translated2.keys()),
        "comment": list(data_translated2.values()),
        "lang": "ru",
    }
)
data_translated2.comment = data_translated2.comment.str.replace('<unk>', '')
data_translated2.comment = data_translated2.comment.str.replace('</s>', '')
data_translated2.comment = data_translated2.comment.str.strip()
data_translated2 = data_translated2[data_translated2.comment.map(len) > 40]
data_translated3 = pd.DataFrame(
    {
        "code": list(data_translated3.keys()),
        "comment": list(data_translated3.values()),
        "lang": "ru",
    }
)
data_translated3.comment = data_translated3.comment.str.replace('<unk>', '')
data_translated3.comment = data_translated3.comment.str.replace('</s>', '')
data_translated3.comment = data_translated3.comment.str.strip()
data_translated3 = data_translated3[data_translated3.comment.map(len) > 40]
data_translated4 = pd.DataFrame(
    {
        "code": list(data_translated4.keys()),
        "comment": list(data_translated4.values()),
        "lang": "ru",
    }
)
data_translated4.comment = data_translated4.comment.str.replace('<unk>', '')
data_translated4.comment = data_translated4.comment.str.replace('</s>', '')
data_translated4.comment = data_translated4.comment.str.strip()
data_translated4 = data_translated4[data_translated4.comment.map(len) > 40]
print(data_translated4.head())
data_translated = pd.concat([data_translated, data_translated2, data_translated3, data_translated4])
data = datasets.Dataset.from_pandas(data_translated)

train_val = data.train_test_split(test_size=0.1, shuffle=True, seed=42)
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

trainer_args = transformers.TrainingArguments(
    per_device_train_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    # gradient_checkpointing=True,
    warmup_steps=100,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    fp16=True,
    # bf16=True,
    tf32=True,
    logging_steps=10,
    evaluation_strategy="steps" if val_set_size > 0 else "no",
    save_strategy="steps",
    eval_steps=400 if val_set_size > 0 else None,
    save_steps=150,
    # fp16_opt_level='O3',
    optim='adamw_torch',
    output_dir=output_dir,
    save_total_limit=1,
    lr_scheduler_type='linear',
    # deepspeed='ds_config_zero.json',
    # load_best_model_at_end=True if val_set_size > 0 else False,
    # ddp_find_unused_parameters=False if ddp else None,
    # group_by_length=group_by_length,
    report_to=None,  # "wandb" if use_wandb else None,
    run_name=None,
    # run_name=wandb_run_name if use_wandb else None,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=trainer_args,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

model = torch.compile(model)
trainer.train()

model.save_pretrained(output_dir)

code = """def generate_and_tokenize_prompt(data_point):
full_prompt = '<lang>' + data_point['lang'] + '<code>' + data_point["code"] + '<docstring>' + data_point["comment"]
tokenized_full_prompt = tokenize(full_prompt)
return tokenized_full_prompt
"""
# full_prompt = prompter.generate_prompt(
#     "Explain this code in "
#     + "Russian. "
#     + "Describe its purpose.",
#     code,
# )
instruct = tokenizer(
    code + "<comment>",
    return_tensors="pt",
    truncation=True,
    max_length=384,
).to("cuda:0")

with torch.inference_mode():
    print(
        tokenizer.decode(
            model.generate(
                input_ids=instruct["input_ids"], num_beams=1, max_new_tokens=256
            )
            .cpu()
            .numpy()[0],
            skip_special_tokens=True
        )
    )

model.save_pretrained(output_dir)