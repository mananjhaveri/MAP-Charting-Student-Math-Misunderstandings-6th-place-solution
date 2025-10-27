# MAP Charting Student Math Misunderstandings - 6th Place Solution Details

> Manan Jhaveri | mananjhaveri.jam@gmail.com
| Data Scientist @ Optum, Mumbai, India

* [Solution on Kaggle Discussion Forum](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/mananjhaveri-map-6th-place-solution)
* [Final submission notebook on Kaggle](https://www.kaggle.com/code/mananjhaveri/map-6th-place-solution-qwen-semble-ftw)


### Hardware Details

* 1 x NVIDIA RTX A5000 24GB
* OS/Platform: Ubuntu 22.04.5 LTS (Jammy Jellyfish)


### How to train the model

* Go to src directory and run the train.py script with the required arguments.
* The arguments include:
    * model_path - HF/Local Model's path to be used as backbone
    * output_folder_name - Name of the folder in which the LoRA adapter, tokenizer and label encoder will be saved after training.
    * learning_rate - Learning rate of the model. Should be changed as per the model backbone (and batch size, etc).
    * use_aug_data - True if you want to use augmented text else False
    * use_synthetic_data - True if you want to use synthetic text else False
    * use_pseudo_labelled_dups - True if you want to use pseudo-labelled duplicates else False
    * debug_mode - True if you want to run in debug on a small subset of the dataset else False
* Use learning rate 1.5e-4 with 8B model and 6e-5 with 14B model.
* Here's an example for finetuning Qwen/Qwen3-Embedding-8B model with synthetic data and pseudo-labelled duplicated samples:
```
! python train.py \
    --model_path Qwen/Qwen3-Embedding-8B \
    --output_folder_name qwen3-emb-8B \
    --learning_rate 1.5e-4 \
    --use_aug_data False \
    --use_synthetic_data True \
    --use_pseudo_labelled_dups True \
    --debug_mode False
```

### How to make predictions on a new test set

* Go to src directory and run the inference.py script with the required arguments.
* The arguments include:
    * The first argument should be the base model's path
    * The second argument should be the name of the folder containing the LoRA adapters.
    * Name of the csv file in which the outputs need to be saved.
* Next, use ensemble.py file. It will read all the generated model predictions file from inference.py script and combine the predictions.
* The final submission file will be saved as submission.csv under the submission dir.
* Here's how to generate the outputs of all the models and ensemble them.
```
! python inference.py \
Qwen/Qwen3-Embedding-8B \
qwen3-8b-emb \
submission_qwen3_emb.csv

! python inference.py \
Qwen/Qwen2.5-14B-Instruct \
qwen2.5-14B-IT \
submission_qwen2_5_14B_IT.csv

! python inference.py \
Qwen/Qwen3-Embedding-8B \
qwen3-8B-emb-aug-data \
submission_qwen3-8B-emb-aug-data.csv

! python inference.py \
Qwen/Qwen3-14B \
qwen3-14B-full-data \
submission_qwen3_14B.csv

! python inference.py \
Qwen/Qwen3-14B \
qwen3-14B-syhntetic-data \
submission_qwen3_14B_synthetic_data.csv

! python inference.py \
Qwen/Qwen3-14B \
map-qwen3-14b-3407 \
submission_qwen3_14B_3407.csv

! python inference.py \
Qwen/Qwen3-Embedding-8B \
qwen3_8b_emb_synthetic_data \
submission_qwen3_emb_synth_data.csv

! python ensemble.py
```


### Key assumptions

* GPU device with at least 24GB VRAM is required to train a model using the give scripts.
* GPU device with at least 16GB VRAM is required to run inference using these models.
* The data directory should have the following files:
    * train.csv
    * test.csv
    * sample_submission.csv
    * duplicate_samples.csv
    * duplicate_samples_pseudo_labels.csv
    * synthetic_data.csv