from transformers import TFAutoModelForCausalLM
from transformers import AutoTokenizer

checkpoint_local = "./saved_dir/"

tokenizer = AutoTokenizer.from_pretrained(checkpoint_local)

prompts = ["乔峰", "段誉", "虚竹"]
inputs = tokenizer(prompts, return_tensors="tf", return_token_type_ids=False)
print(inputs)

model = TFAutoModelForCausalLM.from_pretrained(checkpoint_local)

outputs = model.generate(**inputs, 
                        # max_length=30, 
                         num_beams=5, 
                         do_sample=False)
for one in tokenizer.batch_decode(outputs, skip_special_tokens=True):
    print(one)
