from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from transformers import pipeline
import torch
import datetime

start = datetime.datetime.now()
# pipe = pipeline(model="bigscience/bloom-350m", device_map="auto", max_new_tokens=20, torch_dtype=torch.float16, use_cache=True)
pipe = pipeline(model="bigscience/bloom", device_map="auto", max_new_tokens=20, torch_dtype=torch.bfloat16, use_cache=True)
print(f"Loaded pipeline {datetime.datetime.now() - start}")


out = pipe("test")
print(out)
out = pipe("I enjoy walking my cute dog")
print(out)
out = pipe("Math exercise - answers:\n34+10=44\n54+20=")
print(out)
out = pipe('A "whatpu" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is: We were traveling in Africa and we saw these very cute whatpus. To do a "farduddle" means to jump up and down really fast. An example of a sentence that uses the word farduddle is:')
print(out)
