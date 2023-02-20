from transformers import TFAutoModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./saved_tokenizer/")

prompt = "段誉"
inputs = tokenizer(prompt, return_tensors="tf")
print(inputs)

model = TFAutoModelForCausalLM.from_pretrained("./saved_model_finetuned/")

outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
