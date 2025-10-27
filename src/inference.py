import json
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel, PeftModelForSequenceClassification
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification

import joblib
import gc
import os, sys

import transformers

gc.enable()

settings = json.load(open("./settings.json"))

class TextDataset:

    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.df["text"].iloc[idx],
            # padding="max_length",
            truncation=True,
            max_length=160, # config["max_length"]
        )

        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            # "label": torch.tensor(self.df["label"].iloc[idx])
        }


def format_input(row):
    correctness = "Yes" if row["is_correct"] else "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct: {correctness}\n"
        f"Explanation: {row['StudentExplanation']}\n"
        f"Task: Classify the misconception in the explanation."
    )


def eval_fn(model, dl):
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dl):
            batch = {k: v.to("cuda") for k,v in batch.items()}
    
            out = model(**batch).logits.cpu().detach().tolist()
            predictions.extend(out)
    
    probs = torch.nn.functional.softmax(
        torch.tensor(predictions), dim=1
    ).numpy()
    
    return probs



if __name__ == "__main__":

    base_model_name = sys.argv[1]
    adapter_folder_name = sys.argv[2]
    output_csv_name = sys.argv[3]

    output_path = os.path.join(settings["OUTPUT_DIR"], output_csv_name)
    adapter_path = os.path.join(settings["MODEL_DIR"], adapter_folder_name)

    train = pd.read_csv(os.path.join(settings["RAW_DATA_DIR"], 'train.csv'))

    le = joblib.load(
        os.path.join(adapter_path, "label_encoder.bin")
    )
    
    # train.Misconception = train.Misconception.fillna('NA')
    # train['target'] = train.Category+":"+train.Misconception
    
    train.Misconception = train.Misconception.fillna('NA')
    train['target'] = train.Category.apply(lambda x: x.split("_")[-1]) + ":" + train.Misconception
    
    train['label'] = le.transform(train['target'])
    n_classes = len(le.classes_)
    train.head()
    
    idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
    correct = train.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c',ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])
    correct = correct[['QuestionId','MC_Answer']]
    correct['is_correct'] = 1
    
    train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train.is_correct = train.is_correct.fillna(0)

    train['text'] = train.apply(format_input,axis=1)
    
    test = pd.read_csv(os.path.join(settings["RAW_DATA_DIR"], 'test.csv'))
    test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    test.is_correct = test.is_correct.fillna(0)
    
    test['text'] = test.apply(format_input,axis=1)
    
    test.head()
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    ds = TextDataset(test, tokenizer=tokenizer)

    collate_fn = transformers.DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
    )

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True, 
        bnb_4bit_quant_type = 'nf4',
        bnb_4bit_use_double_quant = True, 
        bnb_4bit_compute_dtype = torch.float16 
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=n_classes,
        quantization_config=quantization_config,
        device_map="auto",
    )
    print("-" * 250)

    model = PeftModelForSequenceClassification.from_pretrained(
        model, 
        adapter_path
    )
    
    model.eval();

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    probs = eval_fn(model, dl)

    test["is_correct"] = test["is_correct"].map({1: "True", 0: "False"})

    top_indices = np.argsort(-probs, axis=1)
    
    flat_indices = top_indices.flatten()
    decoded_labels = le.inverse_transform(flat_indices)
    top_labels = decoded_labels.reshape(top_indices.shape)


    prob_data = []
    for i in range(len(test)):
        prob_dict = {f"prob_{j}": probs[i, top_indices[i, j]] for j in range(37)}
        prob_dict['row_id'] = test.row_id.values[i]

        prefix = test["is_correct"].iloc[i] + "_"
        prob_dict['top_classes'] = " ".join([prefix + label for label in top_labels[i, :37]])
        prob_data.append(prob_dict)
    
    prob_df = pd.DataFrame(prob_data)
    prob_df.to_csv(output_path, index=False)