# %%
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig
import datasets
from tqdm import tqdm
import pickle

# %%
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


def format_system_prompt(prompt, eos_token="</s>"):
    return "{}{}{}{}".format("<|prompter|>", prompt, eos_token, "<|assistant|>")


collator = transformers.DataCollatorWithPadding(
    tokenizer=tokenizer, padding=True, pad_to_multiple_of=8
)


def generate(
    prompt, generation_config=generation_config, max_new_tokens=128, device=device
):
    tokenized = tokenizer(
        [
            format_system_prompt(
                "Explain what this code does. Describe its purpose. " + entry
            )
            for entry in prompt["func_code_string"]
        ],
    )
    collated = collator(tokenized).to(device)
    print(collated["input_ids"].shape)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=collated["input_ids"],
            attention_mask=collated["attention_mask"],
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            eos_token_id=2,
        )
    output = [
        ret.split("<|assistant|>")[-1]
        for ret in tokenizer.batch_decode(generation_output)
    ]
    return output


# %%
data_csn = datasets.load_dataset("martiwey/code-search-net-clean")
# %%
data_csn = data_csn.filter(lambda x: len(x["func_code_string"]) < 304, num_proc=12)
data_csn = data_csn.shuffle(seed=42000)
# %%
model = torch.compile(model)
# %%
batch_size = 8

responses = {}
for i in tqdm(range(0, len(data_csn["train"]), batch_size)):
    output = generate(data_csn["train"][i : i + batch_size])
    for output_idx in range(len(output)):
        responses[data_csn["train"][i + output_idx]["func_code_string"]] = output[
            output_idx
        ]
    if i % 200 == 0:
        with open("llama_comments_7b_final.pkl", "wb") as f:
            pickle.dump(responses, f)
        print("saved responses", i)

with open("llama_comments_7b_final.pkl", "wb") as f:
    pickle.dump(responses, f)

# %%
