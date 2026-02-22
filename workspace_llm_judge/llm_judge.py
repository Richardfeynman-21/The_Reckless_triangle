import os
import google.generativeai as genai

def generate_dynamic_feedback(win_probability, toxicity_severity, chat_logs, player_stats):
    """
    Uses the Gemini LLM as a meta-judge to generate nuanced feedback 
    based on the outputs of our structured + unstructured PyTorch models.
    """
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        print("Failed to initialize Gemini Client. Make sure GEMINI_API_KEY is set in your environment.")
        return str(e)

    prompt = f"""
    You are an expert AI PUBG Coach and Toxicity Moderator. 
    Analyze the current state of your player's match and provide a 2-3 sentence strategic advice and behavioral warning summary.
    
    Current Match Data:
    - Predicted Win Probability (from PyTorch DNN): {win_probability * 100:.1f}%
    - Detected Chat Toxicity Severity (from PyTorch LSTM): {toxicity_severity * 100:.1f}%
    - Player Stats: {player_stats}
    - Recent Team Chat Logs: '{chat_logs}'

    Guidelines:
    - If Win Probability is high but Toxicity is also high, warn them that tilting will throw their lead.
    - If Win Probability is low and Toxicity is high, recommend muting teammates and playing solo survival.
    - If Toxicity is zero, focus purely on tactical advice based on their kills/damage.
    - Keep it strictly formatted as professional coaching advice. Do not output anything else.
    """

    print("--- Asking Gemini to Judge the Match Environment based on PyTorch outputs ---")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {e}"

if __name__ == "__main__":
    # Simulated pipeline output
    test_win_prob = 0.55
    test_toxicity = 0.82
    test_stats = {'kills': 5, 'damage': 1500, 'assists': 2}
    test_chat = "This team is literal garbage, go die."
    
    feedback = generate_dynamic_feedback(test_win_prob, test_toxicity, test_chat, test_stats)
    
    print("\n[LLM GENERATED FEEDBACK]")
    print(feedback)
