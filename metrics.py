import torch
from sentence_transformers import SentenceTransformer, util
import time
from collections import defaultdict

INTENTS = {
    "read_text": [
        "What does this label say?",
        "Can you read the document for me?",
        "Please read out loud what is written here.",
        "Tell me the contents of this sign.",
        "Read the instructions on the box.",
        "What is the text on this page?",
        "Can you tell me what's printed on this flyer?",
        "Read this paragraph to me.",
        "What's written on the notice?",
        "What does this article say?",
        "Can you read the text on the packaging?",
        "What's on this billboard?",
        "Tell me the words on this paper.",
        "What is written on the board?",
        "Read the description on the menu.",
        "What is the content of this pamphlet?",
        "What does the receipt say?",
        "Read the label on this bottle.",
        "Tell me what is on this screen.",
        "What's on this presentation slide?",
        "Can you read the ingredients list?",
        "What is written on this ticket?",
        "Please read what's on the brochure.",
        "What's on this instruction manual?",
        "Can you tell me the content of this message?",
        "Read the front of this card.",
        "What's on this email?",
        "Tell me what's on this poster.",
        "What does this certificate say?",
        "Please read the warning label.",
        "What's on this advertisement?",
        "Read the fine print for me.",
        "Tell me the text on this invitation.",
        "What is written in this announcement?",
        "Read the menu options.",
        "Tell me the headline on this newspaper.",
        "What does this signboard say?",
        "Can you read this chart?",
        "What's on this legal document?",
        "Read the caption under the image.",
        "What is printed on this T-shirt?",
        "Tell me the quote on this mug.",
        "What's written on the wall?",
        "Read the cover of this magazine.",
        "What is this slogan?",
        "Can you read the subtitles?",
        "Tell me what's in this speech bubble.",
        "What does the map legend say?",
        "Read the instructions for assembling this.",
        "What's on the product label?",
        "Can you tell me the tag line?",
        "What does this warning sign say?",
        "Read the notes on this whiteboard.",
        "What's on this application form?",
        "Tell me the birthday card message.",
        "Read the summary on this report.",
        "What is on this payment slip?",
        "Can you tell me the event details?",
        "Read the back of this postcard.",
        "What is on this sign at the store?",
        "Tell me the mission statement here.",
        "What's on this gift tag?",
        "Read the comic strip dialogue.",
        "What does the text on this screen say?",
        "Tell me what’s on this announcement board.",
        "What is on this charity poster?",
        "Read the FAQ section for me.",
        "What’s the title of this book?",
        "Tell me what is on this presentation handout.",
        "What is on this parking sign?",
        "Read the agenda for this meeting.",
        "What does this label on the jar say?",
        "Tell me the information on this package.",
        "What is on this event flyer?",
        "Read the nutritional info on this box.",
        "What's on this instruction leaflet?",
        "Can you read the file name?",
        "Tell me the subject line of this email.",
        "What does this hospital sign say?",
        "Read this conference badge for me.",
        "What is on this user manual cover?",
        "Tell me the writing on this sticky note.",
        "What’s the note on the fridge?",
        "Read this list of items.",
        "What does this chart heading say?",
        "Tell me the product description.",
        "What is the score shown on this screen?",
        "Read the details on this badge.",
        "What is on this banner?",
        "Tell me the words written on this photo.",
        "What is written on this board game card?",
        "Read the certification text.",
        "What’s the brand slogan here?",
        "Tell me what’s printed on this sticker.",
        "Read the customer review shown here.",
        "What is on this discount sign?",
        "Tell me the contents of this sticky label.",
        "What’s on this appointment card?",
        "Read the greeting on this letter."
    ],

    "describe_scene": [
        "What do you see around you?",
        "Can you describe this scene to me?",
        "What is happening here?",
        "Tell me about the surroundings.",
        "What’s in front of us?",
        "Can you explain the environment?",
        "What objects do you notice?",
        "What does the place look like?",
        "Describe the atmosphere here.",
        "What’s going on in this room?",
        "Can you tell me what’s in this photo?",
        "What are people doing here?",
        "How would you describe the setting?",
        "What’s the situation outside?",
        "Tell me what's visible here.",
        "What do you see in this image?",
        "Can you give me an overview of this place?",
        "What are the key elements in this scene?",
        "What’s the general vibe here?",
        "What details can you describe?",
        "Tell me about the background.",
        "What’s on the table?",
        "What’s in this street view?",
        "Describe the landscape.",
        "What kind of activity is happening?",
        "What’s the layout of this place?",
        "What do you notice in the crowd?",
        "Can you describe what the people are doing?",
        "What’s happening behind me?",
        "What are the main colors in this scene?",
        "Tell me about the decor.",
        "What do the buildings look like?",
        "What’s on the floor?",
        "Describe what’s on the shelves.",
        "What kind of weather is it?",
        "What’s on this stage?",
        "What’s the mood of the scene?",
        "What can you see through the window?",
        "Describe the objects on this desk.",
        "What’s hanging on the walls?",
        "What’s happening in the background?",
        "What do you see on this street corner?",
        "Describe the objects near me.",
        "What are the people wearing?",
        "What’s the lighting like?",
        "What’s in the garden?",
        "Describe the scene outside the car.",
        "What’s on this beach?",
        "What’s on this mountain trail?",
        "What do you see in this park?",
        "What’s the setup at this event?",
        "Describe the playground.",
        "What do you see at the bus stop?",
        "What’s on this balcony?",
        "What’s in this alleyway?",
        "What’s on this field?",
        "Describe what’s on the picnic blanket.",
        "What’s in this living room?",
        "What’s the arrangement on the table?",
        "What’s on the kitchen counter?",
        "What’s happening on this sports field?",
        "Describe the activities on the street.",
        "What’s on this rooftop?",
        "What do you see at the fair?",
        "What’s in this backyard?",
        "What’s on the restaurant table?",
        "What’s in this forest scene?",
        "What do you see in this market?",
        "Describe the festival decorations.",
        "What’s in this museum exhibit?",
        "What’s happening on the dance floor?",
        "Describe what’s on the shelf.",
        "What’s the condition of the road?",
        "What’s happening in this shop?",
        "What do you see in the mirror reflection?",
        "What’s in the hallway?",
        "What’s in this office space?",
        "What’s on the construction site?",
        "Describe the farm area.",
        "What’s happening at this bus station?",
        "What’s in this waiting room?",
        "What’s the setup at this concert?",
        "What’s in the airport lounge?",
        "What’s on the sports court?",
        "What’s the view from the balcony?",
        "Describe the public square.",
        "What’s in this hotel lobby?",
        "What’s on this boat deck?",
        "What’s in the car interior?",
        "What do you see in this workshop?",
        "What’s happening on this hiking path?",
        "What’s on the city street at night?",
        "Describe the indoor market scene.",
        "What’s in this flower garden?",
        "What’s on this library table?",
        "What do you see in this aquarium?",
        "What’s in this subway station?",
        "What’s happening in this stadium?"
    ],

    "activate_detection_collision": [
        "Turn on obstacle detection.",
        "Activate collision detection mode.",
        "Please enable collision avoidance.",
        "Start the collision detector.",
        "Switch on obstacle awareness.",
        "Can you enable safety detection?",
        "Turn on the avoidance system.",
        "Activate safety mode.",
        "Please start the collision warning system.",
        "Enable obstacle alert.",
        "Switch to collision detection mode.",
        "Activate the anti-collision feature.",
        "Turn on hazard detection.",
        "Enable object proximity warnings.",
        "Start avoiding obstacles now.",
        "Switch on the danger detection system.",
        "Enable the crash prevention feature.",
        "Activate the safety perimeter mode.",
        "Turn on the obstacle sensor.",
        "Enable auto-stop on collision.",
        "Activate emergency stop system.",
        "Enable obstacle avoidance logic.",
        "Start the collision shield feature.",
        "Activate intelligent safety mode.",
        "Turn on real-time collision checks.",
        "Enable the navigation safety system.",
        "Start monitoring for collisions.",
        "Turn on environmental sensing.",
        "Enable full collision guard.",
        "Activate obstacle monitoring.",
        "Start the protective detection system.",
        "Switch on the smart collision alerts.",
        "Enable advanced collision assistance.",
        "Activate dynamic avoidance.",
        "Turn on the safety scan.",
        "Start barrier detection.",
        "Activate hazard awareness.",
        "Enable automated stopping on impact.",
        "Turn on reactive safety mode.",
        "Enable proximity control.",
        "Start full environment monitoring.",
        "Activate the safe navigation feature.",
        "Enable crash detection sensors.",
        "Switch to emergency obstacle mode.",
        "Start the protective safety net.",
        "Activate the 360-degree detection system.",
        "Turn on multi-directional safety.",
        "Enable intelligent obstacle checks.",
        "Activate the obstacle security system.",
        "Enable path safety checks.",
        "Turn on motion collision prevention.",
        "Activate full awareness mode.",
        "Enable stop-on-detection feature.",
        "Start the anti-impact system.",
        "Activate the obstacle mapping system.",
        "Enable automatic hazard response.",
        "Turn on smart safety barrier.",
        "Activate all-around detection.",
        "Enable live obstacle sensing.",
        "Start object interference detection.",
        "Turn on hazard prevention.",
        "Activate motion interruption feature.",
        "Enable active collision guard.",
        "Start scanning for objects.",
        "Enable area safety monitoring.",
        "Turn on predictive obstacle detection.",
        "Activate obstacle defense mode.",
        "Enable reactive collision checks.",
        "Turn on immediate stop mode.",
        "Enable continuous hazard monitoring.",
        "Start the safety enhancement system.",
        "Activate full coverage scanning.",
        "Enable forward collision alerts.",
        "Turn on backward collision alerts.",
        "Enable lateral safety checks.",
        "Start evasive maneuver mode.",
        "Activate sudden obstacle response.",
        "Enable hazard escape logic.",
        "Turn on all-terrain obstacle mode.",
        "Activate near-field awareness.",
        "Enable adaptive collision system.",
        "Start obstacle hazard mapping.",
        "Turn on predictive safety scans.",
        "Enable object avoidance shield.",
        "Activate ultimate safety mode.",
        "Enable critical zone detection.",
        "Turn on full impact prevention.",
        "Start instant obstacle guard.",
        "Enable auto-evasive feature.",
        "Activate hazard route analysis.",
        "Turn on continuous proximity alerts.",
        "Enable high-sensitivity safety mode.",
        "Start comprehensive collision watch.",
        "Activate next-gen obstacle avoidance.",
        "Enable rapid collision checks.",
        "Turn on real-time object safety.",
        "Enable emergency obstacle watch."
    ],

    "locate_object": [
        "Find my phone.",
        "Where is my wallet?",
        "Locate the keys for me.",
        "Find the remote control.",
        "Where did I leave my glasses?",
        "Can you locate my bag?",
        "Find my shoes.",
        "Where is the laptop?",
        "Locate the water bottle.",
        "Where’s my notebook?",
        "Find my ID card.",
        "Where is my charger?",
        "Locate my headphones.",
        "Find the TV remote.",
        "Where are my socks?",
        "Locate the umbrella.",
        "Where is the passport?",
        "Find my watch.",
        "Locate the camera.",
        "Where’s my hat?",
        "Find my favorite book.",
        "Where is the pen?",
        "Locate my scarf.",
        "Find the flashlight.",
        "Where’s my coffee mug?",
        "Locate the keys to the car.",
        "Find my hairbrush.",
        "Where is the grocery list?",
        "Locate my sneakers.",
        "Find the pencil case.",
        "Where is the USB drive?",
        "Locate the lunch box.",
        "Find my necklace.",
        "Where is the game controller?",
        "Locate my planner.",
        "Find the spare batteries.",
        "Where is the gym bag?",
        "Locate my perfume bottle.",
        "Find the house keys.",
        "Where is my water flask?",
        "Locate the tickets.",
        "Find my reading glasses.",
        "Where are my gloves?",
        "Locate the face mask.",
        "Find my favorite sweater.",
        "Where is the external hard drive?",
        "Locate the grocery bag.",
        "Find my sunscreen.",
        "Where is the suitcase?",
        "Locate my yoga mat.",
        "Find my Bluetooth speaker.",
        "Where is the phone charger?",
        "Locate my favorite pen.",
        "Find the camera lens.",
        "Where are my earphones?",
        "Locate the gift box.",
        "Find my travel pillow.",
        "Where is my planner?",
        "Locate my water jug.",
        "Find the first aid kit.",
        "Where are my keys to the office?",
        "Locate my sleeping bag.",
        "Find the gardening gloves.",
        "Where is my bike helmet?",
        "Locate the laundry basket.",
        "Find my camping gear.",
        "Where is the beach towel?",
        "Locate my passport holder.",
        "Find my slippers.",
        "Where is the cereal box?",
        "Locate my sunglasses.",
        "Find the nail clipper.",
        "Where is my hoodie?",
        "Locate the barbecue tongs.",
        "Find my hiking boots.",
        "Where is the camera tripod?",
        "Locate my favorite blanket.",
        "Find my credit card.",
        "Where is my toolbox?",
        "Locate the picnic basket.",
        "Find my swim goggles.",
        "Where is my cleaning cloth?",
        "Locate my drawing tablet.",
        "Find the toolbox wrench.",
        "Where are my receipts?",
        "Locate my power bank.",
        "Find my bike lock.",
        "Where is the measuring tape?",
        "Locate my reading lamp.",
        "Find the broom.",
        "Where is my diary?",
        "Locate my headphones case.",
        "Find the sewing kit.",
        "Where is my water filter?",
        "Locate my travel guide.",
        "Find the oven mitts.",
        "Where is the floor mop?",
        "Locate my sketchbook."
    ],

    "other": [
        "Tell me a joke.",
        "Sing me a song.",
        "What’s your name?",
        "Who created you?",
        "Do you have feelings?",
        "What time is it?",
        "Can you dance?",
        "What’s your favorite color?",
        "What are you thinking about?",
        "Do you have a family?",
        "What’s your purpose?",
        "Tell me something interesting.",
        "Can you play music?",
        "What do you dream of?",
        "What languages can you speak?",
        "Do you get tired?",
        "Can you draw?",
        "What’s your favorite food?",
        "What do you do for fun?",
        "Do you ever sleep?",
        "Tell me a story.",
        "Do you like animals?",
        "What is love?",
        "Do you like movies?",
        "What’s your favorite song?",
        "What hobbies do you have?",
        "Can you recommend a book?",
        "What’s the weather today?",
        "Do you believe in aliens?",
        "What’s your favorite movie?",
        "Who is your best friend?",
        "Do you like humans?",
        "What makes you happy?",
        "Can you make a poem?",
        "What’s your biggest fear?",
        "Do you believe in ghosts?",
        "What is your favorite sport?",
        "Can you solve a riddle?",
        "Do you play games?",
        "What is the meaning of life?",
        "Can you keep secrets?",
        "What superpower would you want?",
        "What would you do on vacation?",
        "Do you celebrate holidays?",
        "What is your favorite place?",
        "Do you get angry?",
        "What do you think of robots?",
        "What makes you unique?",
        "Can you tell me my future?",
        "Do you get bored?",
        "What do you want to learn?",
        "Who is your hero?",
        "What do you do when alone?",
        "Do you like jokes?",
        "What’s your favorite animal?",
        "What’s your favorite season?",
        "Do you like to travel?",
        "What is your favorite planet?",
        "Do you have dreams?",
        "What’s your favorite day of the week?",
        "What do you wish for?",
        "Can you act like a human?",
        "What would you invent?",
        "What do you think of art?",
        "What would you paint?",
        "Do you like surprises?",
        "What music do you listen to?",
        "What’s your favorite dessert?",
        "What would you do with money?",
        "What gift would you give me?",
        "What’s your favorite drink?",
        "What would you do in a movie?",
        "Can you describe yourself?",
        "What is your favorite quote?",
        "Do you have a secret talent?",
        "What would you do if you were invisible?",
        "What’s your favorite game?",
        "What do you think of humans?",
        "What’s your dream job?",
        "Can you tell me a fun fact?",
        "What would you eat if you could?",
        "What’s your favorite flower?",
        "Can you give me advice?",
        "What do you admire most?",
        "What’s your favorite sound?",
        "What would you change in the world?",
        "What makes you laugh?",
        "What’s your favorite holiday?",
        "What do you want to explore?",
        "What’s your favorite shape?",
        "What do you fear the most?",
        "What would you name a pet?",
        "What do you think of music?",
        "What would you do on an island?",
        "What’s your favorite candy?",
        "What would you do in space?",
        "What do you think of AI?",
        "What is your favorite fruit?"
    ]
}

demand_templates = {
    "read_text": [
            "Can you read aloud what's written here?",
            "Please read the text shown on the screen.",
            "What does the writing say on that sign?",
            "Tell me exactly what the label says.",
            "Could you read the message displayed in front of me?",
            "What are the words written on this surface?",
            "Read this text for me, please."
        ],
        "describe_scene": [
            "Can you describe what's happening around me?",
            "What do you see in this area?",
            "Give me a detailed description of the scene.",
            "Describe the setting and objects nearby.",
            "Tell me what the surroundings look like.",
            "What's visible in the current environment?",
            "What does this room look like?"
        ],
        "activate_detection_collision": [
            "Please enable obstacle and hazard detection.",
            "Turn on the collision prevention system now.",
            "Start the object and movement detection feature.",
            "Activate sensors to detect anything I might bump into.",
            "Can you switch on the obstacle warning system?",
            "Enable collision alerts and monitoring, please."
        ],
        "locate_object": [
            "Where is the phonne?",
            "Can you find my keys?",
            "Help me locate my wallet."
            "Where did I leave my glasses?",
            "Can you track down my backpack?"
        ],
        "other": [
            "Play some background music.",
            "What's the weather forecast for today?",
            "Remind me about my 5 PM meeting.",
            "Call my mother's phone.",
            "Open the phone's camera app.",
            "Show directions to the nearest grocery store."
        ]
}

MODELS = [
    #"all-MiniLM-L6-v2",
    #"all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "all-distilroberta-v1",
    "all-roberta-large-v1",
    #"paraphrase-multilingual-MiniLM-L12-v2",
    #"paraphrase-multilingual-mpnet-base-v2"
]

def build_classifier(model_name):
    model = SentenceTransformer(model_name)
    category_embeddings = {
        intent: torch.mean(model.encode(phrases, convert_to_tensor=True), dim=0)
        for intent, phrases in demand_templates.items()
    }

    def classify(text):
        input_embedding = model.encode(text, convert_to_tensor=True)
        sims = {intent: util.cos_sim(input_embedding, emb).item() for intent, emb in category_embeddings.items()}
        sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_score = sorted_sims[0]
        second_score = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
        # Check if difference is sufficient and minimum score is met
        if top_score - second_score <= 0.05 or top_score < 0.15:
            return "other", top_score - second_score
        return top_intent, top_score

    return classify

def evaluate_classifier(classify_func, dataset):
    correct = 0
    total = 0
    times = []
    incorrect = []
    category_stats = defaultdict(lambda: {
        "correct": 0,
        "total": 0,
        "errors": 0,
        "time": [],
        "misclassified_to": defaultdict(int),
        "misclassified_from": defaultdict(int),
        "misclassified_examples": []
    })

    for true_intent, examples in dataset.items():
        for text in examples:
            start = time.time()
            pred_intent, confidence = classify_func(text)
            end = time.time()
            elapsed = end - start

            category_stats[true_intent]["total"] += 1
            category_stats[true_intent]["time"].append(elapsed)

            if pred_intent == true_intent:
                correct += 1
                category_stats[true_intent]["correct"] += 1
            else:
                incorrect.append((text, true_intent, pred_intent, confidence))
                category_stats[true_intent]["errors"] += 1
                category_stats[true_intent]["misclassified_to"][pred_intent] += 1
                category_stats[pred_intent]["misclassified_from"][true_intent] += 1
                category_stats[true_intent]["misclassified_examples"].append({
                    "text": text,
                    "predicted": pred_intent,
                    "confidence": confidence
                })

            total += 1
            times.append(elapsed)

    accuracy = correct / total
    avg_time = sum(times) / total

    return accuracy, avg_time, incorrect, category_stats

if __name__ == "__main__":
    results = []

    for model_name in MODELS:
        print(f"\nEvaluating model: {model_name}")
        classifier = build_classifier(model_name)
        accuracy, avg_time, errors, stats = evaluate_classifier(classifier, INTENTS)

        print(f"Overall Accuracy: {accuracy*100:.2f}%")
        print(f"Avg Time per Classification: {avg_time*1000:.2f} ms")
        print(f"Total Errors: {len(errors)}")

        print("Per-Category Breakdown:")
        for cat, values in stats.items():
            total_cat = values["total"]
            correct_cat = values["correct"]
            errors_cat = values["errors"]
            avg_cat_time = sum(values["time"]) / total_cat * 1000 if total_cat > 0 else 0.0
            cat_accuracy = correct_cat / total_cat * 100 if total_cat > 0 else 0.0

            misclass_to_str = ", ".join(
                f"{target}: {count}" for target, count in values["misclassified_to"].items()
            ) if values["misclassified_to"] else "None"

            misclass_from_str = ", ".join(
                f"{source}: {count}" for source, count in values["misclassified_from"].items()
            ) if values["misclassified_from"] else "None"

            print(f"  {cat:30s} | Accuracy: {cat_accuracy:5.2f}% | Errors: {errors_cat:3d} | Avg Time: {avg_cat_time:6.2f} ms")
            print(f"      Misclassified to  : {misclass_to_str}")
            print(f"      Misclassified from: {misclass_from_str}")

            # Print each example that was misclassified
            if values["misclassified_examples"]:
                print(f"      Misclassified examples:")
                for ex in values["misclassified_examples"]:
                    print(f"        - \"{ex['text']}\" -> Predicted: {ex['predicted']} (Confidence: {ex['confidence']:.2f})")

        results.append((model_name, accuracy, avg_time, errors, stats))

    # Summary Table
    print("\n--- Summary ---")
    for model_name, acc, t, _, _ in results:
        print(f"{model_name:30s} | Accuracy: {acc*100:.2f}% | Avg Time: {t*1000:.2f} ms")
