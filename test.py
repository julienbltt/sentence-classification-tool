from sbertClassification import classify

if __name__ == "__main__":
    while True:
        text = input("Enter your message: ")

        if text.lower() in ['exit', 'quit', 'q']:
            print("Exiting the program.")
            break

        intent, confidence = classify(text)
        print(f"Predicted Intent: {intent} (confidence={confidence:.2f})")