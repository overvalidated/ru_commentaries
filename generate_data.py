# %%
import pickle

import datasets
import torch
import transformers
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig

# %%
# Some hyperparameters
# make sure
BATCH_SIZE = 8
LORA_WEIGHTS = "jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b"
LLAMA_WEIGHTS = "decapoda-research/llama-7b-hf"
PATH_TO_SAVE = "llama_comments_7b_final.pkl"
DATASET_NAME = "martiwey/code-search-net-clean"


# %%
def format_system_prompt(prompt: str, eos_token: str = "</s>"):
    return "{}{}{}{}".format("<|prompter|>", prompt, eos_token, "<|assistant|>")


def generate(
    prompt: str,
    generation_config: GenerationConfig,
    max_new_tokens: int = 256,
    device: str = "cuda",
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
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=collated["input_ids"],
            attention_mask=collated["attention_mask"],
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            eos_token_id=2,
            pad_token_id=2,
        )
    output = [
        ret.split("<|assistant|>")[-1]
        for ret in tokenizer.batch_decode(generation_output)
    ]
    return output


# %%
# load model, prepare pretrained adapters for prompting
model = transformers.AutoModelForCausalLM.from_pretrained(
    LLAMA_WEIGHTS, device_map="auto", torch_dtype=torch.float16
)  # Load Base Model
model.resize_token_embeddings(
    32005
)  # This model repo also contains several embeddings for special tokens that need to be loaded.

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = transformers.AutoTokenizer.from_pretrained(LORA_WEIGHTS, use_fast=False)
tokenizer.padding_side = "left"

model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model = PeftModel.from_pretrained(
    model,
    LORA_WEIGHTS,
    device_map="auto",
    torch_dtype=torch.float16,
)  # Load Lora model

model.eos_token_id = tokenizer.eos_token_id

# merging lora adapters for inference speed
model.merge_and_unload()
model = model.base_model.model

# %%
# Preloading data
data_csn = datasets.load_dataset(DATASET_NAME)
data_csn = data_csn.filter(lambda x: len(x["func_code_string"]) < 352, num_proc=12)
data_csn = data_csn.shuffle(seed=42000)
# %%

model = torch.compile(model)

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
)

collator = transformers.DataCollatorWithPadding(
    tokenizer=tokenizer, padding=True, pad_to_multiple_of=8
)

responses = {}
for i in tqdm(range(0, len(data_csn), BATCH_SIZE)):
    output = generate(data_csn[i : i + BATCH_SIZE], generation_config=generation_config)
    for output_idx in range(len(output)):
        responses[data_csn[i + output_idx]["func_code_string"]] = output[output_idx]

    # Saving every 400 responses
    if i % 400 == 0:
        with open(PATH_TO_SAVE, "wb") as f:
            pickle.dump(responses, f)
        print("saved responses", i)

with open(PATH_TO_SAVE, "wb") as f:
    pickle.dump(responses, f)
