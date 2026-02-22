import streamlit as st
import numpy as np
import pickle
import os
import torch

from src.models import PUBGGameplayModel, ToxicityLSTMModel
from src.pipeline import PUBGAnalyzer
from llm_judge import generate_dynamic_feedback

st.set_page_config(page_title="PUBG AI Analyzer (PyTorch Engine)", layout="wide")

st.title("PUBG AI Analysis & Feedback System")
st.markdown("Integrates structured gameplay data with unstructured chat data to provide real-time strategic feedback. Powered by PyTorch.")

MODELS_DIR = "models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    st.sidebar.success(f"GPU Acceleration Enabled: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning("GPU Acceleration Disabled. Running on CPU.")

@st.cache_resource
def load_pipeline():
    try:
        scaler = pickle.load(open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb"))
        tokenizer = pickle.load(open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "rb"))
        
        # Load Gameplay Model
        game_model = PUBGGameplayModel(input_shape=scaler.n_features_in_)
        game_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "game_model.pth"), weights_only=True, map_location=DEVICE))
        
        # Load Toxicity Model
        tox_model = ToxicityLSTMModel(vocab_size=tokenizer.vocab_size)
        tox_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "tox_model.pth"), weights_only=True, map_location=DEVICE))
        
        return PUBGAnalyzer(game_model, tox_model, scaler, tokenizer)
    except Exception as e:
        st.warning(f"Could not load trained models: {e}. Are they trained? Run \`python train.py\` first. Proceeding with mocked UI demonstration.")
        return None

analyzer = load_pipeline()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Player Statistics Input")
    st.markdown("Enter simulated gameplay data to evaluate win probability.")
    
    assists = st.number_input("Assists", min_value=0, value=1)
    boosts = st.number_input("Boosts", min_value=0, value=2)
    damage_dealt = st.number_input("Damage Dealt", min_value=0.0, value=150.0)
    dbnos = st.number_input("DBNOs", min_value=0, value=1)
    headshot_kills = st.number_input("Headshot Kills", min_value=0, value=1)
    heals = st.number_input("Heals", min_value=0, value=1)
    kill_place = st.number_input("Kill Place", min_value=1, value=50)
    kill_points = st.number_input("Kill Points", min_value=0, value=1000)
    kills = st.number_input("Kills", min_value=0, value=2)
    kill_streaks = st.number_input("Kill Streaks", min_value=0, value=1)
    longest_kill = st.number_input("Longest Kill (m)", min_value=0.0, value=50.0)
    match_duration = st.number_input("Match Duration (s)", min_value=0, value=1800)
    max_place = st.number_input("Max Place", min_value=1, value=100)
    num_groups = st.number_input("Num Groups", min_value=1, value=25)
    rank_points = st.number_input("Rank Points", min_value=-1, value=-1)
    revives = st.number_input("Revives", min_value=0, value=0)
    ride_distance = st.number_input("Ride Distance", min_value=0.0, value=1000.0)
    road_kills = st.number_input("Road Kills", min_value=0, value=0)
    swim_distance = st.number_input("Swim Distance", min_value=0.0, value=0.0)
    team_kills = st.number_input("Team Kills", min_value=0, value=0)
    vehicle_destroys = st.number_input("Vehicle Destroys", min_value=0, value=0)
    walk_distance = st.number_input("Walk Distance", min_value=0.0, value=2000.0)
    weapons_acquired = st.number_input("Weapons Acquired", min_value=0, value=4)
    win_points = st.number_input("Win Points", min_value=0, value=1500)
    
    stats = {
        "assists": assists, "boosts": boosts, "damageDealt": damage_dealt, "DBNOs": dbnos,
        "headshotKills": headshot_kills, "heals": heals, "killPlace": kill_place,
        "killPoints": kill_points, "kills": kills, "killStreaks": kill_streaks,
        "longestKill": longest_kill, "matchDuration": match_duration, "maxPlace": max_place,
        "numGroups": num_groups, "rankPoints": rank_points, "revives": revives,
        "rideDistance": ride_distance, "roadKills": road_kills, "swimDistance": swim_distance,
        "teamKills": team_kills, "vehicleDestroys": vehicle_destroys, "walkDistance": walk_distance,
        "weaponsAcquired": weapons_acquired, "winPoints": win_points,
        "totalDistance": ride_distance + walk_distance + swim_distance,
        "healsAndBoosts": heals + boosts,
        "headshotRate": headshot_kills / max(1, kills)
    }

with col2:
    st.subheader("Team Chat Analysis")
    st.markdown("Enter simulated recent chat logs from your team.")
    chat_logs = st.text_area("Chat logs (one message per line)", 
                             value="Hey team, drop school?\nCan you drop me an 8x scope?\nOMG you guys are trash, literal bots.")
    
    if st.button("Analyze Match Environment", type="primary"):
        messages = [m.strip() for m in chat_logs.split('\n') if m.strip()]
        
        st.divider()
        st.subheader("Intelligence Report")
        
        if analyzer:
            result = analyzer.analyze(stats, messages)
            
            # OVERRIDE the Python rule-based feedback with the Generative AI Response
            llm_feedback = generate_dynamic_feedback(result['prediction'], result['toxicity_severity'], chat_logs, stats)
            result['feedback'] = llm_feedback
        else:
            toxicity = 0.9 if any(bad in chat_logs.lower() for bad in ['trash', 'bot', 'idiot', 'die']) else 0.1
            flags = ['toxic', 'insult'] if toxicity > 0.5 else []
            win_prob = 0.4 if damage_dealt < 200 else 0.7
            
            fb = []
            if toxicity > 0.8: fb.append("CRITICAL WARNING: Highly toxic communication detected.")
            if win_prob < 0.5: fb.append("TACTICAL ALERT: Odds of early elimination are high.")
            if toxicity > 0.5 and win_prob < 0.5: fb.append("SYSTEM SUGGESTION: Poor synergy. Break off from squad.")
            if not fb: fb.append("STABLE MATCH: Keep looting.")
            
            result = {
                "prediction": win_prob,
                "toxicity_severity": toxicity,
                "toxic_flags": flags,
                "feedback": "\n".join(fb)
            }
            
        colA, colB = st.columns(2)
        with colA:
            st.metric(label="Predicted Win Probability (0 to 1)", value=f"{result['prediction']:.2f}")
            st.progress(float(result['prediction']))
            
        with colB:
            st.metric(label="Toxicity Severity", value=f"{result['toxicity_severity']:.2f}",
                      delta="High Toxicity!" if result['toxicity_severity'] > 0.6 else "Clean Chat", 
                      delta_color="inverse")
            if result['toxic_flags']:
                st.error(f"Detected Tags: {', '.join(result['toxic_flags'])}")
            else:
                st.success("No toxic flags detected.")
                
        st.info(f"**Actionable Feedback:**\n\n{result['feedback']}")
