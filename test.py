from sbertClassification import classify

if __name__ == "__main__":
    while True:
        text = input("Enter your message: ")
        intent, confidence = classify(text)
        print(f"Predicted Intent: {intent} (confidence={confidence:.2f})")