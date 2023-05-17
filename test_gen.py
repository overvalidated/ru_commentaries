# %%
import torch
import transformers
from huggingface_hub import hf_hub_download
from peft import PeftModel

from peft import PeftModel

# %%
##!! CODE FOR LLAMA TESTING
# prompter = Prompter(prompt_template_name)
cutoff_len = 1024
base_model = "decapoda-research/llama-7b-hf"
tokenizer = transformers.LlamaTokenizer.from_pretrained(
    "jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b"
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    torch_dtype=torch.float16,
)  # Load Base Model
model.resize_token_embeddings(
    len(tokenizer)
    # 32016
)  # This model repo also contains several embeddings for special tokens that need to be loaded.

model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

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


def generate_and_tokenize_prompt(data_point):
    full_prompt = (
        "<lang>"
        + data_point["lang"]
        + "<code>"
        + data_point["code"]
        + "<docstring>"
        + data_point["comment"]
    )
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt


# model = prepare_model_for_int8_training(model)
# config = LoraConfig(
#     r=lora_r,
#     lora_alpha=lora_alpha,
#     target_modules=lora_target_modules,
#     lora_dropout=lora_dropout,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
model = PeftModel.from_pretrained(
    model, "./lora-alpaca", torch_dtype=torch.float16
)
model.print_trainable_parameters()

model.eos_token_id = tokenizer.eos_token_id
filename = hf_hub_download(
    "jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b", "extra_embeddings.pt"
)
embed_weights = torch.load(
    filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)  # Load embeddings for special tokens
model.base_model.model.model.embed_tokens.weight[32000:, :] = embed_weights.to(
    model.base_model.model.model.embed_tokens.weight.dtype
).to(
    "cuda"
)  # Add special token embeddings

model = model.to("cuda")
# %%
# model = model.merge_and_unload()
# model.save_pretrained("merged_rucomm_llama")
# model = model.half().to('cuda')
# %%

code = """def generate_and_tokenize_prompt(data_point):
    full_prompt = '<lang>' + data_point['lang'] + '<code>' + data_point["code"] + '<docstring>' + data_point["comment"]
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt
"""

instruct = tokenizer(
    "<lang>en<code>" + code + "<comment>",
    return_tensors="pt",
    truncation=True,
    max_length=cutoff_len,
).to("cuda:0")

with torch.inference_mode():
    print(
        tokenizer.decode(
            model.generate(
                input_ids=instruct["input_ids"], num_beams=1, max_new_tokens=512
            )
            .cpu()
            .numpy()[0],
            skip_special_tokens=True,
        )
    )

# instruct = tokenizer("<lang>ru<code>" + code + "<comment>", return_tensors='pt',
#             truncation=True,
#             max_length=cutoff_len).to('cuda:0')

# with torch.inference_mode():
#     print(tokenizer.decode(
#         model.generate(input_ids=instruct['input_ids'], num_beams=1, max_new_tokens=cutoff_len).cpu().numpy()[0],
#         skip_special_tokens=True))

# %%
##!! CODE FOR SEQ2SEQ TESTING

# tokenizer1 = transformers.AutoTokenizer.from_pretrained("Salesforce/codegen-2B-multi")
# tokenizer1.add_special_tokens({'additional_special_tokens': ['<lang>', '<code>']})
# tokenizer2 = transformers.AutoTokenizer.from_pretrained("ai-forever/rugpt3medium_based_on_gpt2")
tokenizer1 = transformers.AutoTokenizer.from_pretrained(
    "Salesforce/codegen-350M-multi", use_fast=True
)
# tokenizer1.add_special_tokens({'additional_special_tokens': ['<ru>', '<en>']})
tokenizer2 = transformers.GPT2Tokenizer.from_pretrained(
    "ai-forever/rugpt3medium_based_on_gpt2",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    use_fast=True,
)
# tokenizer2.add_special_tokens({'additional_special_tokens': ['<lang>', '<code>']})
# tokenizer1.pad_token_id = 50256#tokenizer1.eos_token_id
# tokenizer2.pad_token_id = 0#tokenizer2.eos_token_id
# tokenizer2.pad_token_id = 0#tokenizer2.eos_token_id
model = transformers.EncoderDecoderModel.from_pretrained(
    "./encoder_decoder/checkpoint-300"
).cuda()
# %%
## Testing for

tokenizer1.padding_side = "left"  # Allow batched inference
tokenizer2.padding_side = "left"  # Allow batched inference
code = """def generate_and_tokenize_prompt(data_point):
    full_prompt = '<lang>' + data_point['lang'] + '<code>' + data_point["code"] + '<docstring>' + data_point["comment"]
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt
"""
# code="""def add(x, y):
# return x+y
# """
model.config.decoder_start_token_id = tokenizer2.bos_token_id  # tokenizer2.bos_token_id
model.config.pad_token_id = tokenizer2.pad_token_id

instruct = tokenizer1(code, return_tensors="pt", truncation=True, max_length=512).to(
    "cuda:0"
)

with torch.inference_mode():
    print(
        tokenizer2.decode(
            model.generate(
                input_ids=instruct["input_ids"], max_new_tokens=64, num_beams=1
            )
            .cpu()
            .numpy()[0],
            skip_special_tokens=True,
        )
    )
# %%
