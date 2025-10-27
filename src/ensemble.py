import json
import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict

settings = json.load(open("./settings.json"))

def extract_class_probabilities(row, model_suffix='', top_k=25):
    """Extract class names and probabilities from a row"""
    # Get top classes
    classes_col = f'top_classes{model_suffix}'
    if classes_col in row:
        classes = row[classes_col].split(' ')[:top_k]
    else:
        return {}
    # Get probabilities
    class_probs = {}
    for i in range(min(top_k, len(classes))):
        prob_col = f'prob_{i}{model_suffix}'
        if prob_col in row:
            class_probs[classes[i]] = row[prob_col]
    return class_probs


def ensemble_with_disagreement_handling(prob_files, model_weights=None, top_k=3):
    n_models = len(prob_files)
    print(n_models)
    prob_dfs = []
    final_predictions = []
    
    for file_path in prob_files:
        df = pd.read_csv(file_path)
        prob_dfs.append(df)
    
    # Merge on row_id
    merged_df = prob_dfs[0]
    for i, df in enumerate(prob_dfs[1:], 1):
        merged_df = pd.merge(merged_df, df, on='row_id', suffixes=('', f'_model{i+1}'))
      
    for idx, row in merged_df.iterrows():
        
        # Extract probabilities from each model
        all_class_probs = []
        for i in range(n_models):
            suffix = f'_model{i+1}' if i > 0 else ''
            class_probs = extract_class_probabilities(row, suffix, top_k=25)
            all_class_probs.append(class_probs)
        
        # Get all unique classes
        all_classes = set()
        for class_probs in all_class_probs:
            all_classes.update(class_probs.keys())
        
        # Calculate agreement and disagreement
        class_votes = defaultdict(int)
        class_total_prob = defaultdict(float)
        class_max_prob = defaultdict(float)
        
        for i, class_probs in enumerate(all_class_probs):
            weight = model_weights[i]
            
            for class_name, prob in class_probs.items():
                class_votes[class_name] += 1
                class_total_prob[class_name] += prob * weight
                class_max_prob[class_name] = max(class_max_prob[class_name], prob * weight)
        
        final_scores = {}
        for class_name in all_classes:
            
            # Base score: weighted average probability
            base_score = class_total_prob[class_name]
            
            # Agreement : classes predicted by more models get boost
            agreement_bonus = class_votes[class_name] / n_models
            
            # Confidence bonus: classes with high max probability get boost
            confidence_bonus = class_max_prob[class_name]
            
            # Combined score
            final_scores[class_name] = (
                base_score * 0.8 +
                agreement_bonus * 0.1 +
                confidence_bonus * 0.1
            )
        
        # Sort and get top-k
        sorted_classes = sorted(final_scores.items(), key=lambda x: -x[1])
        top_classes = [class_name for class_name, _ in sorted_classes[:top_k]]
        
        final_predictions.append(' '.join(top_classes))
    
    return final_predictions


prob_files = glob.glob(settings["OUTPUT_DIR"] + "*.csv")

predictions = ensemble_with_disagreement_handling(
    prob_files, 
    model_weights=[1] * len(prob_files),
    top_k=3
)
    
test = pd.read_csv(os.path.join(settings["RAW_DATA_DIR"], 'test.csv'))
submission = pd.DataFrame({
    'row_id': test.row_id.values,
    'Category:Misconception': predictions
})

submission.to_csv(os.path.join(settings["SUBMISSION_DIR"], 'submission.csv'), index=False)
