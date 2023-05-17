import torch
from transformers import GenerationConfig
import pickle
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm import tqdm

with open("llama_comments.pkl", "rb") as f:
    responses = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-3.3B", torch_dtype=torch.float16, device_map="auto"
)

model = torch.compile(model)

translated_responses = {}
## can be optimized with batching
with torch.inference_mode():
    for n, (code, comment) in enumerate(tqdm(responses.items())):
        inputs = tokenizer(comment, return_tensors="pt").to("cuda:0")
        if len(comment) > 1000:
            continue

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["rus_Cyrl"],
            max_length=1280,
            num_beams=4
        )
        translated_responses[code] = tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )[0]
        if n % 200 == 0:
            with open("llama_comments_translated.pkl", "wb") as f:
                pickle.dump(translated_responses, f)

with open("llama_comments_translated.pkl", "wb") as f:
    pickle.dump(translated_responses, f)
