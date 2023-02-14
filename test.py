
from transformers import TFAutoModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', use_fast = False, tokenize_chinese_chars = True)

prompt = "乔峰道："
inputs = tokenizer(prompt, return_tensors="tf")
print(inputs)


model = TFAutoModelForCausalLM.from_pretrained("gpt2-test")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))