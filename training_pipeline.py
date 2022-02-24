from transformers import  TFAutoModelForSequenceClassification
import tensorflow as tf
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
import os
import pandas as pd
from transformers import TrainingArguments
from transformers import Trainer,DataCollatorWithPadding
import datasets
from sklearn import preprocessing
# download these packages first
# https://anaconda.org/pytorch/pytorch
# https://anaconda.org/conda-forge/tensorflow
# conda install -c huggingface transformers

from datasets import load_dataset
from datasets import Dataset
# from torch.utils.data.dataset import Dataset
from datasets import ClassLabel
import torch


root_path = os.getcwd()

df = load_dataset('csv', data_files=os.path.join(root_path, 'corona_nlp_test.csv'))

max_length = 512

model_name = "roberta-base"

training_args = TrainingArguments("test-trainer")

tokenizer  = AutoTokenizer.from_pretrained(model_name)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model  = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 5)

c2l = ClassLabel(num_classes=5, names=list(set(df['train']['label'])))

def preprocess_function(examples):
    tokenized_batch =  tokenizer(examples["text"], truncation=True, padding = 'max_length' )
    tokenized_batch["label"] = c2l.str2int(examples['label']) 

    return tokenized_batch

df = df.rename_column('Sentiment','labels')
df = df.rename_column('labels','label')
df = df.rename_column('OriginalTweet','text')

tokenized_df = df.map(preprocess_function)

tokenized_df=tokenized_df['train'].train_test_split(.1)

tokenized_df=tokenized_df.remove_columns('UserName')
tokenized_df=tokenized_df.remove_columns('ScreenName')
tokenized_df=tokenized_df.remove_columns('Location')
tokenized_df=tokenized_df.remove_columns('TweetAt')

tokenized_df.set_format(columns = ['input_ids','label','attention_mask'])

small_train_dataset = tokenized_df["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_df["test"].shuffle(seed=42).select(range(1000))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset

    # compute_metrics=compute_metrics,
    # tokenizer=tokenizer,
    # data_collator=data_collator,
)

trainer.train()

trainer.predict(small_eval_dataset)

small_eval_dataset.features

