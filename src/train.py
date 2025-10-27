import os
import json
import random
import pandas as pd
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import transformers
from transformers import (
    AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorWithPadding
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F
import joblib
import argparse
from distutils.util import strtobool


config = {
    "epochs": 3,
    "max_length": 160,
    "batch_size": 12,
    "learning_rate": 1.5e-4,
    
    "use_aug_data": False,
    "use_synthetic_data": True,
    "use_pseudo_labelled_dups": True,

    "train_on_full_data": True,
}

settings = json.load(open("./settings.json"))

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True

    print("Seed set to:", random_seed)

def load_data():
    train = pd.read_csv(os.path.join(settings["RAW_DATA_DIR"], 'train.csv'))
    
    synth_data = pd.read_csv(os.path.join(settings["RAW_DATA_DIR"], 'synthetic_data.csv'))
    dup_dfx = pd.read_csv(os.path.join(settings["RAW_DATA_DIR"], 'duplicate_samples.csv'))
    dups_pseudo_labels = pd.read_csv(os.path.join(settings["RAW_DATA_DIR"], 'duplicate_samples_pseudo_labels.csv'))

    dup_dfx["Category_rect"] = dups_pseudo_labels["Category:Misconception"].apply(lambda x: x.split(" ")[0].split("_", 1)[-1])


    # Find the right value for is_correct for each sample based on frequency of correct answers.
    train.Misconception = train.Misconception.fillna('NA')    
    idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
    correct = train.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c',ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])
    correct = correct[['QuestionId','MC_Answer']]
    correct['is_correct'] = 1
    train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train.is_correct = train.is_correct.fillna(0)
    
    # Prepare final labels using without the True/False prefix and encode them.
    le = LabelEncoder()
    train["Category_rect"] = train["Category"].apply(lambda x: x.split("_")[-1]) 
    train["Category_rect"] = train["Category_rect"] + ":" + train["Misconception"]
    train['label'] = le.fit_transform(train['Category_rect'])
    train["is_correct_text"] = train.is_correct.apply(lambda x: True if x == 1 else False)


    # Drop duplicates with same answer and explanation but different label.
    train["StudentAnswer"] = train["MC_Answer"].astype(str) + "_" + train["StudentExplanation"]
    x = train["StudentAnswer"].value_counts().reset_index()
    dups = x[x["count"] > 1]["StudentAnswer"].tolist()
    dup_df = train[train["StudentAnswer"].isin(dups)]
    dup_label_diff = dup_df.groupby("StudentAnswer")["Category_rect"].unique().reset_index()
    dup_label_diff = dup_label_diff[dup_label_diff["Category_rect"].apply(len) > 1]
    train = train[~train["StudentAnswer"].isin(dup_label_diff["StudentAnswer"].tolist())]

    synth_data = synth_data.merge(train[["QuestionText", "QuestionId"]].drop_duplicates(), on=['QuestionText'], how='left')
    synth_data = synth_data.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    synth_data.is_correct = synth_data.is_correct.fillna(0)
    synth_data["is_correct_text"] = synth_data.is_correct.apply(lambda x: True if x == 1 else False)
    synth_data["label"] = le.fit_transform(synth_data['Category_rect'])

    dup_dfx = dup_dfx.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    dup_dfx.is_correct = dup_dfx.is_correct.fillna(0)
    dup_dfx["is_correct_text"] = dup_dfx.is_correct.apply(lambda x: True if x == 1 else False)
    dup_dfx["label"] = le.fit_transform(dup_dfx['Category_rect'])

    joblib.dump(le, os.path.join(config["output_dir"], "label_encoder.bin"))

    return train, synth_data, dup_dfx



def generate_aug_text(text):
    if '+' in text:
        return text.replace("+", " plus ")
    if 'multiply' in text:
        return text.replace("multiply", "times")
    if "=" in text:
        return text.replace("=", " equals ")
    if "divided" in text:
        return text.replace("divided", "divuded")
    if "times" in text:
        return text.replace("times", "multiply")
    if "," in text:
        return text.replace(",", " ")
    if text.endswith("."): 
        return text[: -1]
    else: 
        return text + "."

def custom_sample(df, N):
    sampled_df = []
    for cat in df["Category_rect"].unique():
        sub_df = df[df["Category_rect"] == cat]
        sub_df = sub_df.sample(min(N, sub_df.shape[0]), random_state=42)
        sampled_df.append(sub_df)

    return pd.concat(sampled_df, axis=0)

def get_aug_samples(train, samples_per_category=50):        
    questionwise_options = dict(
        train.groupby("QuestionId")["MC_Answer"].apply(lambda x: x.unique().tolist())
    )
    
    aug_df = []
    
    for idx, row in train_df.iterrows():
        for opt in  questionwise_options[row["QuestionId"]]:
            if opt == row["MC_Answer"]:
                continue    
            new_row = dict(row)
            new_row["MC_Answer"] = opt
            new_row["is_correct"] = 1 if row["is_correct"] == 0 else 0
            aug_df.append(new_row)
    
    
    aug_df = pd.DataFrame(aug_df)
    
    aug_df["StudentExplanation"] = aug_df["StudentExplanation"].apply(generate_aug_text)
    aug_df = custom_sample(aug_df, N=samples_per_category)
    
    return aug_df

def format_input(row):
    correctness = "Yes" if row["is_correct"] else "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct: {correctness}\n"
        f"Explanation: {row['StudentExplanation']}\n"
        f"Task: Classify the misconception in the explanation."
    )

def prepare_ds(train_df, val_df, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=160) # padding="max_length"
        
    COLS = ['text','label']
    train_ds = Dataset.from_pandas(train_df[COLS])
    val_ds = Dataset.from_pandas(val_df[COLS])
    
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    
    columns = ['input_ids', 'attention_mask', 'label']
    train_ds.set_format(type='torch', columns=columns)
    val_ds.set_format(type='torch', columns=columns)

    return train_ds, val_ds
    

def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    map1 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
            map1 += 1
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3

    return {
        "map1": map1 / len(labels),
        "map3": map3 / len(labels),
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path")
    parser.add_argument("--output_folder_name")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--use_aug_data")
    parser.add_argument("--use_synthetic_data",)
    parser.add_argument("--use_pseudo_labelled_dups")
    parser.add_argument("--debug_mode")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    set_random_seed(42 if not args.seed else args.seed)

    config["model_path"] = args.model_path
    config["output_dir"] = os.path.join(settings["MODEL_DIR"], args.output_folder_name)
    config["learning_rate"] = args.learning_rate
    config["use_aug_data"] = strtobool(args.use_aug_data)
    config["use_synthetic_data"] = strtobool(args.use_synthetic_data)
    config["use_pseudo_labelled_dups"] = strtobool(args.use_pseudo_labelled_dups)
    config["debug"] = strtobool(args.debug_mode)

    os.mkdir(config["output_dir"])
    
    train, synth_df, pseudo_labelled_dups_df = load_data()

    train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

    additional_data = []
    if config["use_aug_data"]:
        additional_data.append(get_aug_samples(train))
    if config["use_synthetic_data"]:
        additional_data.append(synth_df)
    if config["use_pseudo_labelled_dups"]:
        additional_data.append(pseudo_labelled_dups_df)

    if config["train_on_full_data"]:
        train_df = pd.concat([train_df, val_df])
    
    train_df = pd.concat(additional_data + [train_df], axis=0)
        
    train_df['text'] = train_df.apply(format_input,axis=1)
    val_df['text'] = val_df.apply(format_input,axis=1)
    config["num_labels"] = train_df["label"].nunique()
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

    if config["debug"]:
        train_df = train_df.sample(1000)
        val_df = val_df.sample(1000)
    
    train_ds, val_ds = prepare_ds(train_df, val_df, tokenizer)
    print("Prepared dataset!")


    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True, 
        bnb_4bit_quant_type = 'nf4',
        bnb_4bit_use_double_quant = True, 
        bnb_4bit_compute_dtype = torch.bfloat16 
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_path"],
        num_labels=config["num_labels"],
        quantization_config=quantization_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=128,
        lora_alpha=32,
        lora_dropout = 0.05,
        task_type = 'SEQ_CLS',
        bias="none",
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1    
    model.print_trainable_parameters()
    print("Model loaded!")
    
    collate_fn = transformers.DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt"
    )

    training_args = TrainingArguments(
        output_dir = config["output_dir"],
    
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
    
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        warmup_ratio=0.05,
    
        save_total_limit=2,
    
        metric_for_best_model="map3",
        greater_is_better=True,
        load_best_model_at_end=True,
    
        report_to="none",
    
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_map3,
        data_collator=collate_fn,
    )
    
    trainer.train()
    trainer.save_model(DIR + "/best")
