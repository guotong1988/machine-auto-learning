from transformers import AutoTokenizer, TFAutoModel
import transformers

print(transformers.__version__)

from datasets import load_dataset

input_data = load_dataset("data", data_files="all.txt")
for i in range(1, 10):
    print(input_data["train"][i])

tokenizer_checkpoint = "sgugger/gpt2-like-tokenizer"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = input_data.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

for i in range(1, 10):
    print(tokenized_datasets["train"][i])

block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=100,
    num_proc=4,
)

for i in range(1, 10):
    print(tokenizer.decode(lm_datasets["train"][i]["input_ids"]))

print()
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = TFAutoModel.from_pretrained("gpt2")
#
# inputs = tokenizer("Hello world!", return_tensors="tf")
# outputs = model(**inputs)
# print(inputs)
# print(outputs)
