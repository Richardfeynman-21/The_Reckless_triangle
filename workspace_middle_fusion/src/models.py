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
        pooled = torch.max(lstm_out, dim=1)[0] # shape: [batch_size, hidden_dim * 2]
        
        logits = self.fc(pooled) # shape: [batch_size, num_classes]
        return logits, pooled # Return pooled for the Middle-Fusion approach

class UnifiedPUBGModel(nn.Module):
    """
    Middle-Fusion approach: concatenates Gameplay DNN embeddings and Toxicity LSTM embeddings
    before feeding them into a final prediction head.
    """
    def __init__(self, game_input_shape, vocab_size=20001, embed_dim=128, hidden_dim=64, num_action_categories=1):
        super(UnifiedPUBGModel, self).__init__()
        
        # Instantiate sub-models
        self.game_model = PUBGGameplayModel(game_input_shape)
        self.text_model = ToxicityLSTMModel(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
        
        # Extract embeddings instead of final predictions by bypassing their last layer
        self.game_encoder = nn.Sequential(*list(self.game_model.network.children())[:-1])
        
        # The joint classifier takes the 32 from game_encoder and (hidden_dim * 2) from text_encoder
        self.joint_classifier = nn.Sequential(
            nn.Linear(32 + (hidden_dim * 2), 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_action_categories) # E.g., predicting exact winPlacePerc (regression)
        )

    def forward(self, stats, text_indices):
        game_features = self.game_encoder(stats) # Shape: [batch, 32]
        
        # Text model returns (logits, pooled_embeddings)
        _, text_features = self.text_model(text_indices) # Shape: [batch, 128 (hidden*2)]
        
        # Concatenate horizontally
        combined = torch.cat((game_features, text_features), dim=1) # Shape: [batch, 160]
        
        return self.joint_classifier(combined)

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
