from transformers import AutoTokenizer, TFAutoModel
import transformers

from transformers import create_optimizer

import tensorflow as tf

print(transformers.__version__)

from datasets import load_dataset

input_data = load_dataset("data", data_files="tianlongbabu.txt")
for i in range(1, 10):
    print(input_data["train"][i])

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    tokenize_chinese_chars=True)

tokenizer.save_pretrained("./tokenizer/")


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

from transformers import AutoConfig, TFAutoModelForCausalLM

model_checkpoint = "gpt2"

config = AutoConfig.from_pretrained(model_checkpoint)

config.save_pretrained("./saved_model/")

model = TFAutoModelForCausalLM.from_config(config)

from transformers import TFTrainer, TFTrainingArguments

training_args = TFTrainingArguments(
    f"{model_checkpoint}-test",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    num_train_epochs=3
)

num_replicas = training_args.strategy.num_replicas_in_sync
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

tf_train_dataset = model.prepare_tf_dataset(
    lm_datasets["train"],
    shuffle=True,
    batch_size=num_replicas * training_args.per_device_train_batch_size,
).with_options(options)

tf_eval_dataset = model.prepare_tf_dataset(
    lm_datasets["train"],
    shuffle=False,
    batch_size=num_replicas * training_args.per_device_eval_batch_size,
    drop_remainder=True,
).with_options(options)

num_train_steps = len(tf_train_dataset) * int(training_args.num_train_epochs)
num_warmup_steps = 0

optimizer, lr_schedule = create_optimizer(
    init_lr=training_args.learning_rate,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    adam_beta1=training_args.adam_beta1,
    adam_beta2=training_args.adam_beta2,
    adam_epsilon=training_args.adam_epsilon,
    weight_decay_rate=training_args.weight_decay,
    adam_global_clipnorm=training_args.max_grad_norm,
)

# no user-specified loss = will use the model internal loss
model.compile(optimizer=optimizer)
callbacks = []
history = model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    epochs=int(training_args.num_train_epochs),
    callbacks=callbacks,
)
train_loss = history.history["loss"][-1]
print(train_loss)

validation_loss = history.history["val_loss"][-1]
print(validation_loss)
model.save_pretrained(training_args.output_dir)
