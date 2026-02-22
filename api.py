import os
import torch
import warnings
import numpy as np
import pickle
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from workspace_middle_fusion.src.models import UnifiedPUBGModel

warnings.filterwarnings('ignore')

app = FastAPI(title="PUBG AI Analyzer API")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODELS_DIR = "models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unified_model = None
scaler = None
tokenizer = None

# Configure Gemini
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    gemini_model = None

# Bootstrapping models on startup
@app.on_event("startup")
def load_models():
    global unified_model, scaler, tokenizer
    try:
        # We'll use the models stored in the root models directory
        fusion_dir = "models"
        
        scaler = pickle.load(open(os.path.join(fusion_dir, "scaler.pkl"), "rb"))
        tokenizer = pickle.load(open(os.path.join(fusion_dir, "tokenizer.pkl"), "rb"))
        
        unified_model = UnifiedPUBGModel(game_input_shape=scaler.n_features_in_, vocab_size=tokenizer.vocab_size)
        unified_model.load_state_dict(torch.load(os.path.join(fusion_dir, "joint_fusion_model.pth"), weights_only=True, map_location=DEVICE))
        unified_model.to(DEVICE)
        unified_model.eval()
        print(f"Models loaded successfully on {DEVICE}")
    except Exception as e:
        print(f"Failed to load models: {e}")

class MatchPayload(BaseModel):
    stats: dict
    chatLogs: List[str]

@app.post("/api/analyze")
async def analyze_match(payload: MatchPayload):
    if not unified_model or not scaler or not tokenizer:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    try:
        ordered_keys = [
            'assists', 'boosts', 'damageDealt', 'DBNOs',
            'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
            'killStreaks', 'longestKill', 'matchDuration', 'maxPlace',
            'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills',
            'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
            'weaponsAcquired', 'winPoints', 'totalDistance', 'healsAndBoosts', 'headshotRate'
        ]
        
        # Calculate derived features if not present
        stats = payload.stats
        if 'totalDistance' not in stats:
            stats['totalDistance'] = stats.get('rideDistance', 0) + stats.get('walkDistance', 0) + stats.get('swimDistance', 0)
        if 'healsAndBoosts' not in stats:
            stats['healsAndBoosts'] = stats.get('heals', 0) + stats.get('boosts', 0)
        if 'headshotRate' not in stats:
            kills = max(1, stats.get('kills', 1))
            stats['headshotRate'] = stats.get('headshotKills', 0) / kills

        # 1. Prepare Game Stats
        x_num = np.array([[stats.get(k, 0) for k in ordered_keys]])
        x_scaled = scaler.transform(x_num)
        
        # 2. Prepare Chat Text
        chat_text = " ".join(payload.chatLogs)
        if not chat_text.strip():
            chat_text = "clean"
        x_text = tokenizer.transform([chat_text])
        
        # 3. Middle Fusion Inference
        with torch.no_grad():
            pred = unified_model(
                torch.tensor(x_scaled, dtype=torch.float32).to(DEVICE),
                torch.tensor(x_text, dtype=torch.long).to(DEVICE)
            )
            
        win_prob = float(pred.cpu().numpy()[0][0])
        win_prob = max(0.0, min(1.0, win_prob))
        
        # Approximate toxicity heuristic for dashboard UI visual usage (since Joint Model merges it)
        toxicity_severity = 0.9 if any(bad in chat_text.lower() for bad in ['trash', 'bot', 'idiot', 'die', 'fuck']) else 0.05
        
        # 4. Generate LLM Feedback + Playstyle Profiler Badge
        # If Gemini is offline, we output the pure Middle Fusion PyTorch results.
        persona = "Neural Network Enabled"
        coach_feedback = f"The PyTorch Middle Fusion Architecture predicts a {win_prob*100:.1f}% chance of winning, dynamically fusing your raw structured telemetry stats and unstructured team chat embeddings into a single joint prediction layer."
        
        if gemini_model:
            try:
                prompt = f"""
                You are the PUBG "Playstyle Profiler" and AI Coach. 
                Analyze these inputs:
                - Model Predicted Win Probability: {win_prob * 100:.1f}%
                - Recent Team Chat: '{chat_text}'
                - Player Stats: {stats}

                Respond strictly with a JSON object containing exactly two keys:
                1. "persona": A 1-2 word cool gaming badge describing their playstyle based on the stats (e.g. "Aggressive Rusher", "Tactical Medic", "Passive Survivor", "Toxic Thrower").
                2. "feedback": A 2 sentence strategic coaching advice on what to do next in the match based on their win probability and team chat synergy.
                """
                response = gemini_model.generate_content(prompt)
                
                # Simple extraction of JSON
                import json
                text = response.text.replace('```json', '').replace('```', '').strip()
                data = json.loads(text)
                coach_feedback = data.get("feedback", coach_feedback)
                persona = data.get("persona", persona)
            except Exception as e:
                print("Gemini generation failed: ", e)
                
        return {
            "prediction": win_prob,
            "toxicity": toxicity_severity,
            "persona": persona,
            "feedback": coach_feedback
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
