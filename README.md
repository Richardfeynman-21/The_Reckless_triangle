# The Reckless Triangle - PUBG AI Match Analyzer

Welcome to the **PUBG AI Analyzer & Toxicity Detection System**. This project is a full-stack, machine-learning-powered web application that acts as an "AI Coach," evaluating a player's in-game statistics alongside their team's chat logs to accurately predict their probability of winning using a **Middle Fusion PyTorch Neural Network**.

![Home Page Presentation](docs/home_page.png)

## Core Features
1. **Middle Fusion Architecture:** Instead of evaluating game stats and text sentiment separately, this project utilizes a custom PyTorch model that dynamically fuses numerical telemetry embeddings with NLP chat embeddings into a joint prediction layer for superior accuracy.
2. **Gameplay Outcome Predictor:** A Deep Neural Network that regresses 24 unique features (like Kills, Damage, Heals, longestKill) tracked during a match to predict the final placement tier. Achieves an impressive `0.075` Mean Absolute Error (MAE) and `0.89` R-Squared (R2) score.
3. **Toxicity NLP Engine:** A bidirectional LSTM neural network trained to flag highly toxic team chat communications (Toxic, Severe Toxic, Obscene, Threat, Insult). This acts as a real-time behavioral input into the overall win-chance algorithm.
4. **Agentic Diagnostics:** Fallback heuristics to seamlessly parse raw network outputs into English-readable Playstyle Personas when external Generative AI keys are missing.
5. **Premium Glassmorphic UI:** A state-of-the-art React frontend featuring cinematic animations, smooth CSS transitions, and interactive stat toggles designed directly from raw aesthetic engineering.

---

## 📸 Dashboard Interface
The Live AI Dashboard provides real-time telemetry updates. The player directly interacts with the AI Engine to review their expected win rating and toxicity penalty:
![Live Dashboard Interface](docs/dashboard.png)

---

## 🏆 Hackathon Brownie Points Achieved
- [x] **Unified Multi-Modal Integration:** Successfully merged Unstructured Textual Data (Chat Toxicity) and Structured Tabular Data (Telemetry) into one seamless analytical pipeline.
- [x] **Ethics & Responsible AI:** Addressed the dangers of NLP false-positives by prioritizing supportive UI-level behavioral warnings and dynamically adjusting win-probabilities rather than enforcing harsh punitive server bans.
- [x] **Interactive Dashboard Prototype:** Delivered a fully functional, live-inference web application that goes far beyond a static Jupyter Notebook.
- [x] **Generative AI Integration:** Leveraged Google Gemini (GenAI) via API to override raw numerical outputs with personalized, conversational coaching advice.

---

## 🌟 Extra Features (Beyond the Problem Statement)
To make **The Reckless Triangle** a truly premium product, we engineered several features not requested in the original prompt:
* **Middle-Fusion PyTorch Architecture:** Instead of a simple "rule-based" `if/else` wrapper combining two separate models, we mathematically fused the NLP `nn.LSTM` embeddings and Telemetry features into a single, joint deep-learning prediction layer (`joint_fusion_model.pth`).
* **Playstyle Profiler:** A heuristic sub-engine that dynamically categorizes players into tactical personas (e.g., "Aggressive Rusher", "Tactical Medic") based on their normalized stats map.
* **Premium Glassmorphic Engineering:** Replaced standard Streamlit/Gradio with a custom-engineered **Vite + React** Single Page Application featuring advanced CSS micro-animations, glass-shine hovers, and 4K custom BGMI aesthetic backgrounds.
* **Advanced Dynamic Telemetry Toggles:** Engineered an expandable metric grid allowing power-users to manually inject and analyze all 24 hidden PyTorch features without cluttering the main UI.

---

## 🔬 Model Performance & Loss Metrics

### 1. Gameplay Outcome Predictor (Structured Regression)
* **Architecture:** Deep Neural Network featuring `nn.Linear`, `nn.BatchNorm1d`, and `nn.Dropout(0.3)`.
* **Loss Function:** `nn.MSELoss()` (Mean Squared Error). Chosen to heavily penalize large prediction variances safely scaling the final percentage.
* **Testing MAE:** `0.075` (Our model's predictions are, on average, within 7.5% of the player's true final match placement).
* **Testing R-Squared (R²):** `0.894` (The model successfully explains ~89.4% of the mathematical variance in match outcomes).

### 2. Toxicity Detection Engine (NLP Multi-Label Classification)
* **Architecture:** Bidirectional `nn.LSTM` processing sequences from a custom native `nn.Embedding` vocabulary layer.
* **Loss Function:** `nn.BCEWithLogitsLoss()` (Binary Cross-Entropy with Logits). Crucial for this model as it evaluates 6 independent multi-label probabilities (e.g., a message can simultaneously be "Toxic" and an "Insult" without softmax interference).
* **Macro ROC-AUC:** `0.924` (Demonstrates excellent probabilistic distinction between highly toxic and non-toxic baselines despite severe Jigsaw dataset class imbalance).
* **Macro F1-Score:** `0.254` (Reflects the strict thresholding required due to the massive 90% non-toxic class imbalance, ensuring we minimize false positives).

---

## 🚀 Quickstart Guide

To run this application locally, you will need to start both the PyTorch inference server (Backend) and the React application (Frontend).

### 1. Backend Setup (FastAPI + PyTorch)
Ensure you have Python installed, navigate to the root directory, and set up your virtual environment:
```bash
python -m venv venv
venv\Scripts\activate   # Or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

If you wish to enable the **Gemini Playstyle Coach** (optional), create a `.env` file in the root directory and add your key:
```env
GEMINI_API_KEY=your_key_here
```

Start the Uvicorn server:
```bash
python -m uvicorn api:app --reload --port 8000
```


### 2. Frontend Setup (React + Vite)
Open a new terminal session, navigate to the `frontend` directory, and install the Node dependencies:
```bash
cd frontend
npm install
npm run dev -- --port 5173 --host
```

Open a web browser and navigate to `http://localhost:5173/` to interact with the application.

---

## 🧠 Neural Network Pipeline Details

Our data pipeline integrates two massive public datasets (PUBG PLacement and Jigsaw Toxicity) dynamically. 
1. **Numerical Stream:** Raw game stats are normalized via `StandardScaler`.
2. **Text Stream:** Unstructured chat is tokenized, vectorized, and compressed using a specialized `nn.Embedding` -> `nn.LSTM` pipeline.
3. **Fusion Flow:** The concatenated tensor drives the final `nn.Linear` fully-connected network to yield the ultimate real-time win probability `winPlacePerc`.

*Note: All core model weights (`.pth`), scalers, and tokenizers are serialized and bundled directly in the `/models` directory for immediate local inference—no extra downloads required!*

---
**Developed by The Reckless Triangle**
