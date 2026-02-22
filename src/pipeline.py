import torch
import numpy as np

class PUBGAnalyzer:
    """
    Unified Intelligence Pipeline for PUBG using PyTorch.
    Integrates the Toxicity NLP model and the Gameplay Outcome model.
    """
    def __init__(self, game_model, tox_model, scaler, tokenizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.game_model = game_model.to(self.device)
        self.game_model.eval()
        
        self.tox_model = tox_model.to(self.device)
        self.tox_model.eval()
        
        self.scaler = scaler
        self.tokenizer = tokenizer
        
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def analyze(self, gameplay_stats, chat_messages):
        # 1. Gameplay Prediction
        features_array = np.array([list(gameplay_stats.values())])
        scaled_features = self.scaler.transform(features_array)
        
        with torch.no_grad():
            x_game = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
            win_place_pred = self.game_model(x_game).cpu().numpy()[0][0]
            # Constrain to 0-1 for regression outputs naturally
            win_place_pred = max(0.0, min(1.0, float(win_place_pred)))
        
        # 2. Toxicity Detection
        toxic_flags = []
        max_severity = 0.0
        
        if chat_messages:
            # Clean and Tokenize
            from src.data import clean_text
            cleaned_msgs = [clean_text(m) for m in chat_messages]
            vec_messages = self.tokenizer.transform(cleaned_msgs)
            
            with torch.no_grad():
                x_tox = torch.tensor(vec_messages, dtype=torch.long).to(self.device)
                logits = self.tox_model(x_tox)
                # Apply sigmoid since BCEWithLogitsLoss was used for training
                probs = torch.sigmoid(logits).cpu().numpy()
                
            # Aggregate severity across all messages
            max_preds = probs.max(axis=0) # max score per category
            max_severity = float(max_preds.max()) # overall highest severity
            
            for i, score in enumerate(max_preds):
                if score > 0.5:
                    toxic_flags.append(self.toxicity_labels[i])
                    
        # 3. Contextual Feedback Generation
        feedback = self._generate_feedback(win_place_pred, max_severity, toxic_flags)
        
        return {
            "prediction": win_place_pred,
            "toxicity_severity": max_severity,
            "toxic_flags": toxic_flags,
            "feedback": feedback
        }

    def _generate_feedback(self, win_place_pred, max_severity, toxic_flags):
        feedback = []
        
        if max_severity > 0.8:
            feedback.append("CRITICAL WARNING: Highly toxic communication detected. Proceeding to auto-mute toxic teammates.")
        elif max_severity > 0.5:
            feedback.append("WARNING: Hostile chat environment. Focus on gameplay and ignore negativity.")
            
        if win_place_pred < 0.3:
            feedback.append("TACTICAL ALERT: Odds of early elimination are high based on current stats.")
        elif win_place_pred > 0.8:
            feedback.append("EXCELLENT PACE: You are on track for a Top 10 finish. Stick to rotation strategy.")
        else:
            feedback.append("STABLE MATCH: Keep looting and stick with the safe zone edge.")
            
        if max_severity > 0.5 and win_place_pred < 0.4:
            feedback.append("SYSTEM SUGGESTION: Poor synergy + High Toxicity = High Death Risk. Consider breaking off from the squad.")
            
        return " \n".join(feedback)
