from transformers import TFAutoModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./saved_tokenizer/")

prompts = ["乔峰", "段誉", "虚竹"]
inputs = tokenizer(prompts, return_tensors="tf")
print(inputs)

model = TFAutoModelForCausalLM.from_pretrained("./saved_model_finetuned/")

outputs = model.generate(**inputs, max_length=30, top_k=2, do_sample=False)
for one in tokenizer.batch_decode(outputs, skip_special_tokens=True):
    print(one)
