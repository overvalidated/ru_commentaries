# %%s
import pickle

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding

# %%
BATCH_SIZE = 4
TRANSLATION_MODEL = "facebook/nllb-200-3.3B"
COMMENTS_DATA = "llama_comments_7b_final_2.pkl"
PATH_TO_SAVE = "llama_comments_7b_final_translated_2.pkl"

# %%
# Loading ready english comments
with open(COMMENTS_DATA, "rb") as f:
    responses = pickle.load(f)
# %%
tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL, torch_dtype=torch.float16, device_map="auto"
)
# %%
code = list(responses.keys())
comments = [entry.replace("</s>", "").strip() for entry in list(responses.values())]
# %%
model = torch.compile(model)
tokenizer.padding_side = "left"
# %%
translated_responses = {}
collator = DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8)
## can be optimized with batching
with torch.inference_mode():
    for n in tqdm(range(0, len(comments), BATCH_SIZE)):
        inputs = tokenizer(comments[n : n + BATCH_SIZE], return_tensors=None)
        collated = collator(inputs).to("cuda:0")

        translated_tokens = model.generate(
            **collated,
            forced_bos_token_id=tokenizer.lang_code_to_id["rus_Cyrl"],
            max_length=256,
            num_beams=4
        )
        decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        for entry in range(len(translated_tokens)):
            translated_responses[n + entry] = decoded[entry]

        if n % 200 == 0:
            with open(PATH_TO_SAVE, "wb") as f:
                pickle.dump(translated_responses, f)

with open(PATH_TO_SAVE, "wb") as f:
    pickle.dump(translated_responses, f)

# %%
