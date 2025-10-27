# Training

1. Qwen3 8B Embedding
```
! python train.py \
    --model_path Qwen/Qwen3-Embedding-8B \
    --output_folder_name qwen3_emb_8B \
    --learning_rate 1.5e-4 \
    --use_aug_data False \
    --use_synthetic_data False \
    --use_pseudo_labelled_dups False \
    --debug_mode False
```

2. Qwen3 8B Embedding with augmented data
```
! python train.py \
    --model_path Qwen/Qwen3-Embedding-8B \
    --output_folder_name qwen3_emb_8B_aug_data \
    --learning_rate 1.5e-4 \
    --use_aug_data True \
    --use_synthetic_data False \
    --use_pseudo_labelled_dups False \
    --debug_mode False
```

3. Qwen3 8B Embedding with synthetic data and pseudo-labelled duplicated samples
```
! python train.py \
    --model_path Qwen/Qwen3-Embedding-8B \
    --output_folder_name qwen3_emb_8B_synth_data \
    --learning_rate 1.5e-4 \
    --use_aug_data False \
    --use_synthetic_data True \
    --use_pseudo_labelled_dups True \
    --debug_mode False
```

4. Qwen3 14B
```
! python train.py \
    --model_path Qwen/Qwen3-14B \
    --output_folder_name qwen3_14B \
    --learning_rate 6e-5 \
    --use_aug_data False \
    --use_synthetic_data False \
    --use_pseudo_labelled_dups False \
    --debug_mode False
```

5. Qwen3 14B with synthetic data and pseudo-labelled duplicated samples
```
! python train.py \
    --model_path Qwen/Qwen3-14B \
    --output_folder_name qwen3_14B_synth_data \
    --learning_rate 6e-5 \
    --use_aug_data False \
    --use_synthetic_data True \
    --use_pseudo_labelled_dups True \
    --debug_mode False
```

5. Qwen3 14B with synthetic data and pseudo-labelled duplicated samples, diff seed
```
! python train.py \
    --model_path Qwen/Qwen3-14B \
    --output_folder_name qwen3_14B_synth_data_3407 \
    --learning_rate 6e-5 \
    --use_aug_data False \
    --use_synthetic_data True \
    --use_pseudo_labelled_dups True \
    --debug_mode False \
    --seed 3407
```

6. Qwen2.5 14B IT
```
! python train.py \
    --model_path Qwen/Qwen2.5-14B-Instruct \
    --output_folder_name qwen2_5_14B_IT \
    --learning_rate 6e-5 \
    --use_aug_data False \
    --use_synthetic_data False \
    --use_pseudo_labelled_dups False \
    --debug_mode False
```



# Inference
```
! python inference.py \
Qwen/Qwen3-Embedding-8B \
qwen3_emb_8B \
submission_qwen3_emb.csv

! python inference.py \
Qwen/Qwen2.5-14B-Instruct \
qwen2_5_14B_IT \
submission_qwen2_5_14B_IT.csv

! python inference.py \
Qwen/Qwen3-Embedding-8B \
qwen3_emb_8B_aug_data \
submission_qwen3-8B-emb-aug-data.csv

! python inference.py \
Qwen/Qwen3-14B \
qwen3_14B \
submission_qwen3_14B.csv

! python inference.py \
Qwen/Qwen3-14B \
qwen3_14B_synth_data \
submission_qwen3_14B_synthetic_data.csv

! python inference.py \
Qwen/Qwen3-14B \
qwen3_14B_synth_data_3407 \
submission_qwen3_14B_3407.csv

! python inference.py \
Qwen/Qwen3-Embedding-8B \
qwen3_emb_8B_synth_data \
submission_qwen3_emb_synth_data.csv
```

# Ensemble
```
! python ensemble.py
```