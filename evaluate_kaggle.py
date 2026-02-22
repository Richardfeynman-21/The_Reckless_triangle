import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from src.data import clean_text
from src.models import ToxicityLSTMModel

MODELS_DIR = "models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval_jigsaw():
    print(f"--- Evaluating Toxicity Model on Kaggle Test Set ({DEVICE}) ---")
    
    test_data = pd.read_csv(r"f:\noobathon.bgmi\archive\test.csv")
    test_labels = pd.read_csv(r"f:\noobathon.bgmi\archive\test_labels.csv")
    
    # Merge on id
    df = pd.merge(test_data, test_labels, on='id')
    
    # Kaggle marks un-scored test samples with -1 in the labels. Drop these.
    df = df[(df['toxic'] != -1) & (df['severe_toxic'] != -1)]
    
    print(f"Total valid test samples: {len(df)}")
    
    # Clean text
    df['comment_text'] = df['comment_text'].fillna("").apply(clean_text)
    X = df['comment_text'].values
    y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
    
    # Load tokenizer and model
    with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
        
    X_vec = tokenizer.transform(X)
    
    dataset = TensorDataset(torch.tensor(X_vec, dtype=torch.long), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    model = ToxicityLSTMModel(vocab_size=tokenizer.vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "tox_model.pth"), weights_only=True, map_location=DEVICE))
    model.eval()
    
    all_preds_probs = []
    
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds_probs.append(probs)
            
    preds_probs = np.vstack(all_preds_probs)
    preds_binary = (preds_probs > 0.5).astype(int)
    
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    print("\n[ Classification Report ]")
    print(classification_report(y, preds_binary, target_names=labels, zero_division=0))
    
    print("\n[ ROC-AUC Scores ]")
    for i, label in enumerate(labels):
        auc = roc_auc_score(y[:, i], preds_probs[:, i])
        print(f"{label.ljust(15)}: {auc:.4f}")

if __name__ == "__main__":
    eval_jigsaw()
    print("\nNote: The PUBG Kaggle `test_V2.csv` cannot be locally evaluated because it contains NO labeled ground truth (`winPlacePerc`). Kaggle requires submission of predictions to their server to score it. The 0.07 MAE previously reported was obtained via a strict 80/20 train/test holdout split on the labeled `train_V2.csv` file.")
