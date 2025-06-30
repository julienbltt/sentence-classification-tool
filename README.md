# Sentence Classification Tool

This tool provides real-time classification of natural language questions for seamless integration into the [**Commpanion Blind**](https://github.com/julienbltt/commpanion-blind) project. It uses [Sentence-BERT](https://www.sbert.net/) to semantically match user input to pre-defined intent categories.

---

## Features

- Classifies questions into intent categories using SBERT (`all-distilroberta-v1`)
- Fast and lightweight for real-time interaction
- Easily extendable with additional categories
- Built-in CLI for quick testing

---

## Intents & Templates

Currently supports classification into the following categories:

### `read_text`
Example prompts:
- "Can you read aloud what’s written here?"
- "Please read the text shown on the screen."
- "What does the writing say on that sign?"
- "Tell me exactly what the label says."
- "Could you read the message displayed in front of me?"
- "What are the words written on this surface?"

### `describe_scene`
Example prompts:
- "Can you describe what’s happening around me?"
- "What do you see in this area?"
- "Give me a detailed description of the scene."
- "Describe the setting and objects nearby."
- "Tell me what the surroundings look like."
- "What’s visible in the current environment?"

### `activate_detection_collision`
Example prompts:
- "Please enable obstacle and hazard detection."
- "Turn on the collision prevention system now."
- "Start the object and movement detection feature."
- "Activate sensors to detect anything I might bump into."
- "Can you switch on the obstacle warning system?"
- "Enable collision alerts and monitoring, please."

### `other`
Example prompts:
- "Play some background music."
- "What’s the weather forecast for today?"
- "Remind me about my 5 PM meeting."
- "Call my mother’s phone."
- "Open the phone’s camera app."
- "Show directions to the nearest grocery store."

---

## Getting Started

> **Note:** This tool was developed and tested on **Python 3.10**. Compatibility with other Python versions has not been tested.

### 1. Install dependencies

```bash
pip install sentence-transformers torch
```

### 2. Run the demo

```bash
python test.py
```

You’ll be prompted to enter a message. The system will return the predicted intent and a confidence score.

## Usage Example

The function take in input the intent (str) and will return the predicted intent category (str) and the confidence (float).

```python
from sbertClassification import classify

text = "can you read this for me?"
intent, confidence = classify(text)
print(f"Predicted Intent: {intent} (confidence={confidence:.2f})")
```
Output:
```yaml
Predicted Intent: read_text (confidence=0.53)
```

## How does it work

1) Input text is embedded using SBERT (`all-distilroberta-v1`).
2) Each intent category has a pre-computed mean embedding from its example templates.
3) The tool computes cosine similarity between the input and each intent category.
4) The most similar intent is returned along with a confidence score.

## Performance Metrics

Evaluated on a dataset of 100 samples per category (300 total):
| Model                | Accuracy | Avg Time (ms) |
| -------------------- | -------- | ------------- |
| all-distilroberta-v1 | 99.67%   | 19.47         |

The classifier demonstrates high accuracy and low inference latency, making it suitable for real-time systems