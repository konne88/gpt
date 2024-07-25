# from llm import autocomplete  -- use this import instead of the next one if you don't want key-value chaching
from llmkv import autocomplete
from llama.model import llama
from llama.tokenizer import Tokenizer
from llama.params import llama7BParams, llama1BParams

max_seq_len = 100
tokenizer = Tokenizer(model_path="../llama-2-7b-chat/tokenizer.model")
transformer = llama(llama7BParams(), max_seq_len)
prompt = "[INST] What's 1+1? [/INST]"

tokens = tokenizer.encode(prompt, bos=True, eos=False)
for token in autocomplete(transformer, max_seq_len, tokens):
    if (token == tokenizer.eos_id):
        break
    print(tokenizer.decode([token]), end=' ', flush=True)
