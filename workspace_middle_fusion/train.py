import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.data import load_and_preprocess_pubg, load_and_preprocess_jigsaw
from src.models import UnifiedPUBGModel

MODELS_DIR = "models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_middle_fusion():
    print(f"--- Training Joint PyTorch UnifiedPUBGModel on {DEVICE} ---")
    
    # 1. Load Data
    print("Loading datasets...")
    # NOTE: Since Jigsaw and PUBG datasets share no common keys, we will
    # synthetically align a subset just to demonstrate Middle Fusion training mathematically.
    # In reality, you'd match PlayerID to PlayerChatLog ID.
    
    X_train_g, X_test_g, y_train_g, y_test_g, scaler = load_and_preprocess_pubg(sample_size=10000)
    X_train_t, X_test_t, _, _, tokenizer = load_and_preprocess_jigsaw(sample_size=10000)
    
    # Take the intersection length
    train_len = min(len(X_train_g), len(X_train_t))
    test_len = min(len(X_test_g), len(X_test_t))
    
    X_train_g, X_train_t, y_train = X_train_g[:train_len], X_train_t[:train_len], y_train_g[:train_len]
    X_test_g, X_test_t, y_test = X_test_g[:test_len], X_test_t[:test_len], y_test_g[:test_len]
    
    # 2. Prepare DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train_g, dtype=torch.float32), 
        torch.tensor(X_train_t, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 3. Initialize Model
    model = UnifiedPUBGModel(
        game_input_shape=X_train_g.shape[1], 
        vocab_size=tokenizer.vocab_size,
        num_action_categories=1 # Regression for winPlacePerc
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Train
    epochs = 4
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for g_batch, t_batch, y_batch in train_loader:
            g_batch, t_batch, y_batch = g_batch.to(DEVICE), t_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(g_batch, t_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Joint MSE Loss: {total_loss/len(train_loader):.4f}")
        
    # 5. Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(
            torch.tensor(X_test_g, dtype=torch.float32).to(DEVICE),
            torch.tensor(X_test_t, dtype=torch.long).to(DEVICE)
        ).cpu().numpy().flatten()
    
    mae = mean_absolute_error(y_test, preds)
    print(f">> Joint Middle-Fusion MAE Validation Score: {mae:.4f}")
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "joint_fusion_model.pth"))
    print("Joint model saved successfully.\n")

if __name__ == "__main__":
    train_middle_fusion()
