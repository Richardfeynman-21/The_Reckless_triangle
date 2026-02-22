import torch
import torch.nn as nn

class PUBGGameplayModel(nn.Module):
    """
    DNN for predicting PUBG match outcome (winPlacePerc Regression).
    """
    def __init__(self, input_shape):
        super(PUBGGameplayModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output prob/rank 0.0 to 1.0 (linear, to be constrained by MSE)
        )

    def forward(self, x):
        return self.network(x)

class PUBGGameplayClassificationModel(nn.Module):
    """
    DNN for predicting PUBG match outcome (Classification into 4 bins).
    """
    def __init__(self, input_shape, num_classes=4):
        super(PUBGGameplayClassificationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes) # Output logits for CrossEntropyLoss
        )

    def forward(self, x):
        return self.network(x)

class ToxicityLSTMModel(nn.Module):
    """
    NLP multi-label classification model for toxicity detection using PyTorch.
    """
    def __init__(self, vocab_size=20001, embed_dim=128, hidden_dim=64, num_classes=6):
        super(ToxicityLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Bidirectional LSTM -> hidden config is hidden_dim * 2
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes) 
            # Sigmoid is applied during loss (BCEWithLogitsLoss) or at inference
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embeds = self.embedding(x) # [batch_size, seq_len, embed_dim]
        lstm_out, (hidden, cell) = self.lstm(embeds) # lstm_out: [batch_size, seq_len, hidden_dim * 2]
        
        # Global Max Pooling 1D over seq_len
        # Max pool over the sequence dimension (dim=1)
        pooled = torch.max(lstm_out, dim=1)[0] # shape: [batch_size, hidden_dim * 2]
        
        logits = self.fc(pooled) # shape: [batch_size, num_classes]
        return logits

def test_models():
    # Test gameplay model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dummy_game_in = torch.randn(8, 20).to(device)
    game_model = PUBGGameplayModel(20).to(device)
    game_out = game_model(dummy_game_in)
    print("Game Model Out Shape:", game_out.shape)

    dummy_tox_in = torch.randint(0, 1000, (8, 100)).to(device)
    tox_model = ToxicityLSTMModel().to(device)
    tox_out = tox_model(dummy_tox_in)
    print("Toxicity Model Out Shape:", tox_out.shape)

if __name__ == "__main__":
    test_models()
