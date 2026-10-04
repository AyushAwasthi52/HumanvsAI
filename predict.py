import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    parser = argparse.ArgumentParser(description="Predict if text is Human or AI generated")
    parser.add_argument("--model", type=str, default="roberta-base-openai-detector", 
                        help="Path to custom model dir or Hugging Face model ID")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    # For custom models we trained (0=Human, 1=AI). 
    # For roberta-base-openai-detector (0=Fake/AI, 1=Real/Human).
    parser.add_argument("--is_custom", action="store_true", 
                        help="Flag to indicate if this is our custom trained model (flips label mapping)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model and tokenizer: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("If using a custom model, ensure you've trained it first and are pointing to the correct directory (e.g., ./custom_detector)")
        return
        
    model.to(device)
    model.eval()
    
    inputs = tokenizer(args.text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().tolist()
        pred_idx = torch.argmax(logits, dim=-1).item()
        
    # Mapping
    if args.is_custom:
        # Our custom training script: 0 -> Human, 1 -> AI
        labels = ["Human", "AI"]
        human_prob = probabilities[0]
        ai_prob = probabilities[1]
    else:
        # roberta-base-openai-detector: 0 -> AI (Fake), 1 -> Human (Real)
        labels = ["AI", "Human"]
        # re-align for output
        ai_prob = probabilities[0]
        human_prob = probabilities[1]
        # map pred_idx to our standard 0=Human, 1=AI
        pred_idx = 1 if pred_idx == 0 else 0
        
    predicted_label = "AI" if pred_idx == 1 else "Human"
    
    print("\n--- Prediction Results ---")
    print(f"Input Text: \"{args.text[:100]}...\"" if len(args.text) > 100 else f"Input Text: \"{args.text}\"")
    print(f"Prediction: {predicted_label} Generated")
    print(f"Confidence (Human): {human_prob:.2%}")
    print(f"Confidence (AI):    {ai_prob:.2%}")

if __name__ == "__main__":
    main()
