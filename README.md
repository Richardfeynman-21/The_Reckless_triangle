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
