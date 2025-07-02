from intent_classifier import IntentClassifier

if __name__ == "__main__":
    # Initialize the classifier
    print("🚀 Initializing Intent Classifier...")
    classifier = IntentClassifier()
    
    print("💬 Intent Classification System Ready!")
    print("Available commands: 'exit', 'quit', 'q' to stop, 'help' for more options")
    print("-" * 60)
    
    try:
        while True:
            text = input("Enter your message: ")
            
            if text.lower() in ['exit', 'quit', 'q']:
                print("👋 Exiting the program.")
                break
            
            if text.lower() == 'help':
                print("\n📋 Available commands:")
                print("  - 'exit', 'quit', 'q': Exit the program")
                print("  - 'help': Show this help message")
                print("  - 'categories': Show available intent categories")
                print("  - 'details <your text>': Get detailed classification analysis")
                print("  - Any other text: Classify the intent")
                print()
                continue
            
            if text.lower() == 'categories':
                categories = classifier.get_intent_categories()
                print(f"\n📂 Available categories ({len(categories)}):")
                for i, category in enumerate(categories, 1):
                    examples = classifier.get_category_examples(category)
                    print(f"  {i}. {category} ({len(examples)} examples)")
                print()
                continue
            
            if text.lower().startswith('details '):
                analysis_text = text[8:]  # Remove 'details ' prefix
                if analysis_text.strip():
                    details = classifier.get_classification_details(analysis_text)
                    print(f"\n🔍 Detailed Analysis:")
                    print(f"  Input: '{details.get('input_text', 'N/A')}'")
                    print(f"  Predicted Intent: {details.get('predicted_intent', 'N/A')}")
                    print(f"  Confidence: {details.get('confidence', 0):.4f}")
                    print(f"  All Scores:")
                    for intent, score in details.get('all_scores', {}).items():
                        print(f"    - {intent}: {score:.4f}")
                    print()
                else:
                    print("⚠️ Please provide text after 'details' command")
                continue
            
            if not text.strip():
                print("⚠️ Please enter some text to classify")
                continue
            
            # Classify the text
            intent, confidence = classifier.classify(text)
            print(f"🎯 Predicted Intent: {intent} (confidence={confidence:.4f})")
    
    except KeyboardInterrupt:
        print("\n\n⚡ Program interrupted by user")
    
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    
    finally:
        # Clean up resources
        print("🧹 Cleaning up resources...")
        classifier.cleanup()
        print("✅ Program terminated successfully")