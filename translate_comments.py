import torch
from transformers import GenerationConfig
import pickle
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding
import torch
from tqdm import tqdm

with open("llama_comments_7b_final.pkl", "rb") as f:
    responses = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-3.3B", torch_dtype=torch.float16, device_map='auto'
)
code = list(responses.keys())
comments = [entry.replace("</s>", '').strip() for entry in list(responses.values())]

model = torch.compile(model)
tokenizer.padding_side='left'

translated_responses = {}
batch_size = 4
collator = DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8)
## can be optimized with batching
with torch.inference_mode():
    for n in tqdm(range(0, len(comments), batch_size)):
        inputs = tokenizer(comments[n:n+batch_size], return_tensors=None)
        collated = collator(inputs).to('cuda:0')

        translated_tokens = model.generate(
            **collated,
            forced_bos_token_id=tokenizer.lang_code_to_id["rus_Cyrl"],
            max_length=256,
            num_beams=4
        )
        decoded = tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )
        for entry in range(len(translated_tokens)):
            translated_responses[n+entry] = decoded[entry]

        if n % 200 == 0:
            with open("llama_comments_7b_final_translated.pkl", "wb") as f:
                pickle.dump(translated_responses, f)

with open("llama_comments_7b_final_translated.pkl", "wb") as f:
    pickle.dump(translated_responses, f)
