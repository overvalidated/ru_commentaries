# %%
from typing import List, NamedTuple

import torch
import transformers
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import GenerationConfig
import deepspeed
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import load_checkpoint_and_dispatch
from accelerate import Accelerator
import datasets


model = transformers.AutoModelForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf", device_map="auto", torch_dtype=torch.float16
)  # Load Base Model
model.resize_token_embeddings(
    32005
)  # This model repo also contains several embeddings for special tokens that need to be loaded.

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b", use_fast=False
)

model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

lora_weights = "jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b"
model = PeftModel.from_pretrained(
    model,
    lora_weights,
    device_map="auto",
    torch_dtype=torch.float16,
)  # Load Lora model
model.eos_token_id = tokenizer.eos_token_id
model.merge_and_unload()
model = model.base_model.model
# %%
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
)

# model = torch.compile(model, backend='nvprims_nvfuser')

def format_system_prompt(prompt, eos_token="</s>"):
    return "{}{}{}{}".format("<|prompter|>", prompt, eos_token, "<|assistant|>")


def generate(
    prompt, generation_config=generation_config, max_new_tokens=384, device=device
):
    prompt = format_system_prompt(prompt)  # OpenAssistant Prompt Format expected
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            eos_token_id=2,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s).split("<|assistant|>")[-1]
    return output


# %%
data_csn = datasets.load_dataset("martiwey/code-search-net-clean").shuffle(seed=42)
# %%
data_csn = data_csn.filter(lambda x: len(x["func_code_string"]) < 768, num_proc=12)
data_csn = data_csn.shuffle(seed=42)
# %%
model = torch.compile(model)
# %%
from tqdm import tqdm
import pickle

responses = {}
for n, i in enumerate(tqdm(data_csn["train"])):
    output = generate(
        "Explain what this code does. Describe its purpose. " + i["func_code_string"]
    )
    responses[i["func_code_string"]] = output
    if n % 200 == 0:
        with open("llama_comments_7b_final.pkl", "wb") as f:
            pickle.dump(responses, f)
        print("saved responses", n)

with open("llama_comments_7b_final.pkl", "wb") as f:
    pickle.dump(responses, f)
