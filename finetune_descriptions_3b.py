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
from tqdm import tqdm
import evaluate
from utils.prompter import Prompter


output_dir: str = "./lora-alpaca-3b"
num_epochs: int = 2
prompt_template_name: str = "openassistant"  # The prompt template to use, will default to openassistant.
batch_size = 256
micro_batch_size = 1
learning_rate = 2e-4
cutoff_len = 256+256
eval_mode = False
gradient_accumulation_steps = batch_size // micro_batch_size

prompter = Prompter(prompt_template_name)
tokenizer = transformers.LlamaTokenizer.from_pretrained(
    "openlm-research/open_llama_3b_350bt_preview"
)
model = transformers.LlamaForCausalLM.from_pretrained(
    "openlm-research/open_llama_3b_350bt_preview", 
    torch_dtype=torch.float16
)  # Load Base Model

tokenizer.pad_token_id = 0
tokenizer.bad_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.padding_side = "left"  # Allow batched inference

def tokenize(prompt, add_eos_token=True, tensors=None):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt, truncation=True, max_length=cutoff_len, return_tensors=tensors
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
    full_prompt = data_point['code'] + "<comment>" + data_point['comment']
    tokenized_full_prompt = tokenize(full_prompt)
    user_prompt = data_point['code'] + "<comment>"
    tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ]  # could be sped up, probably
    return tokenized_full_prompt

if not eval_mode:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
else:
    model = PeftModel.from_pretrained(model, output_dir, is_trainable=False)
    model.eval()

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
    warmup_steps=100,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    fp16=True,
    tf32=True,
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=400,
    save_steps=100,
    max_steps=400,
    optim='adamw_torch',
    output_dir=output_dir,
    save_total_limit=1,
    lr_scheduler_type='linear',
    report_to=None,
    run_name=None,
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

sacrebleu = evaluate.load('sacrebleu')
rouge = evaluate.load('rouge')

with torch.inference_mode():
    gts = []
    preds = []
    for batch in tqdm(train_val["test"].shuffle(seed=42)):
        full_prompt = batch['code'] + "<comment>"
        tokenized_full_prompt = tokenizer(
            full_prompt, truncation=True, max_length=cutoff_len, return_tensors='pt'
        ).to('cuda:0')

        predict = tokenizer.decode(
            model.generate(
                input_ids=tokenized_full_prompt["input_ids"], num_beams=1, max_new_tokens=256
            )
            .cpu()
            .numpy()[0],
            skip_special_tokens=True
        ).split('<comment>')[-1]
        preds.append(predict)
        gts.append([batch['comment']])
    print('3b blue', sacrebleu.compute(predictions=preds, references=gts)['score'])
    print('3b rouge2', rouge.compute(predictions=preds, references=gts))
