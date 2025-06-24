from sentence_transformers import SentenceTransformer, util
import torch

# Load and setup the model
model = SentenceTransformer('all-MiniLM-L12-v2') 

demand_templates = {
    "read_text": [
        "Can you read aloud what’s written here?",
        "Please read the text shown on the screen.",
        "What does the writing say on that sign?",
        "Tell me exactly what the label says.",
        "Could you read the message displayed in front of me?",
        "What are the words written on this surface?"
    ],
    "describe_scene": [
        "Can you describe what’s happening around me?",
        "What do you see in this area?",
        "Give me a detailed description of the scene.",
        "Describe the setting and objects nearby.",
        "Tell me what the surroundings look like.",
        "What’s visible in the current environment?"
    ],
    "activate_detection_collision": [
        "Please enable obstacle and hazard detection.",
        "Turn on the collision prevention system now.",
        "Start the object and movement detection feature.",
        "Activate sensors to detect anything I might bump into.",
        "Can you switch on the obstacle warning system?",
        "Enable collision alerts and monitoring, please."
    ],
    "other": [
        "Play some background music.",
        "What’s the weather forecast for today?",
        "Remind me about my 5 PM meeting.",
        "Call my mother’s phone.",
        "Open the phone’s camera app.",
        "Show directions to the nearest grocery store."
    ]
}

category_embeddings = {
    intent: torch.mean(model.encode(phrases, convert_to_tensor=True), dim=0)
    for intent, phrases in demand_templates.items()
}

# Function call to use the tool
def classify(text, margin_threshold=0.1):
    """
    Classify input text into an intent category using SBERT embeddings.
    If the top two category similarities are too close, classify as 'other'.

    Parameters:
        text (str): The input sentence to classify.
        margin_threshold (float): Minimum required difference between top two scores.

    Returns:
        tuple[str, float]: Predicted intent or 'other', and its confidence score.
    """
    user_embedding = model.encode(text, convert_to_tensor=True)
    similarities = {
        intent: util.cos_sim(user_embedding, emb).item()
        for intent, emb in category_embeddings.items()
    }

    sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_intent, top_score = sorted_sims[0]
    second_score = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0

    if top_score - second_score <= margin_threshold or top_score < 0.2:
        return "other", top_score - second_score

    return top_intent, top_score
