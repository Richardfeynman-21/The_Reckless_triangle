import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, accuracy_score

from src.data import load_and_preprocess_pubg, load_and_preprocess_pubg_classification, load_and_preprocess_jigsaw
from src.models import PUBGGameplayModel, PUBGGameplayClassificationModel, ToxicityLSTMModel

MODELS_DIR = "models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_regression(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        preds = model(X_test_tensor).cpu().numpy().flatten()
    mae = mean_absolute_error(y_test, preds)
    return mae

def train_game_model_regression():
    print(f"--- Training PUBG Regression Model (winPlacePerc) on {DEVICE} ---")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_pubg(sample_size=100000)
    
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    model = PUBGGameplayModel(input_shape=X_train.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - MSE Loss: {total_loss/len(train_loader):.4f}")
        
    mae = evaluate_regression(model, X_test, y_test)
    print(f">> Regression Test MAE: {mae:.4f}")
    
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "game_model.pth"))
    print("Gameplay regression model saved successfully.\n")
    return mae

def evaluate_classification(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = model(X_test_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
    f1 = f1_score(y_test, preds, average='weighted')
    acc = accuracy_score(y_test, preds)
    return acc, f1

def train_game_model_classification():
    print(f"--- Training PUBG Classification Model (4 Bins) on {DEVICE} ---")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_pubg_classification(sample_size=100000)
        
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    model = PUBGGameplayClassificationModel(input_shape=X_train.shape[1], num_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - CE Loss: {total_loss/len(train_loader):.4f}")
        
    acc, f1 = evaluate_classification(model, X_test, y_test)
    print(f">> Classification Test Accuracy: {acc:.4f} | F1-Score: {f1:.4f}\n")
    return f1

def train_tox_model():
    print(f"--- Training Toxicity Model on {DEVICE} ---")
    X_train, X_test, y_train, y_test, tokenizer = load_and_preprocess_jigsaw(sample_size=50000)
    
    with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
        
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = ToxicityLSTMModel(vocab_size=tokenizer.vocab_size).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - BCE Loss: {total_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "tox_model.pth"))
    print("Toxicity model saved successfully.\n")


def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    try:
        mae = train_game_model_regression()
        f1 = train_game_model_classification()
        
        print(f"=== MODEL COMPARISON ===")
        print(f"Regression MAE (Mean Absolute Error, lower is better): {mae:.4f}")
        print(f"Classification F1-Score (higher is better): {f1:.4f}")
        print("Note: Regression predicts exact placement (0.0 to 1.0) and translates directly to probability risk.")
        print("Classification bins players into 4 discrete buckets. For a dashboard feedback generator, Regression bounded continuous outputs often provide more nuanced risk dials, while classification might be 'easier' to train for general categories.")
        print("We will keep the Regression model for `app.py` as it preserves granular match placement percentiles.")
        print("========================\n")
    except Exception as e:
        print(f"Error training PUBG model: {e}")

    try:
        train_tox_model()
    except Exception as e:
        print(f"Error training Toxicity model: {e}")

if __name__ == "__main__":
    main()
