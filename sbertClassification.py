from sentence_transformers import SentenceTransformer, util
import torch

# Load and setup the model
model = SentenceTransformer('all-distilroberta-v1') 

demand_templates = { # 6 examples per category to avoid "over fitting"
    "read_text": [
        "tell me what does it say",
        "read the text",
        "can you read this",
        "what's written here",
        "what is written on this menu",
        "Read the instructions for me",
        "tell me the words on the sign"
    ],
    "describe_scene": [
        "what do you see",
        "cam you explain me what's around me",
        "describe me the scene",
        "what's happening here",
        "can you tell me what is in front of me",
        "describe the surroundings"
    ],
    "activate_detection_collision": [
        "Turn on the collision detector.",
        "Activate collision detection.",
        "Can you enable obstacle detection?",
        "Start the collision avoidance system.",
        "Please switch on collision detection.",
        "Enable the object detection feature."
    ]
}

category_embeddings = {
    intent: torch.mean(model.encode(phrases, convert_to_tensor=True), dim=0)
    for intent, phrases in demand_templates.items()
}

# Function call to use the tool
def classify(text):
    """
    Classify input text into an intent category using SBERT embeddings.

    Parameters:
        text (str): The input sentence to classify.

    Returns:
        tuple[str, float]: A tuple containing the predicted intent category (as a string)
                           and the confidence score (cosine similarity) as a float.
    """
    user_embedding = model.encode(text, convert_to_tensor=True)
    similarities = {
        intent: util.cos_sim(user_embedding, emb).item()
        for intent, emb in category_embeddings.items()
    }
    best_intent = max(similarities, key=similarities.get)
    confidence = similarities[best_intent]
    return best_intent, confidence