import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm
import pandas as pd

def load_hc3_dataset(split='train', max_samples=1000):
    print(f"Loading Hello-SimpleAI/HC3 dataset (all.jsonl)...")
    # Load directly from jsonl as the loading script is deprecated
    dataset = load_dataset("json", data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl", split="train")
    
    texts = []
    labels = []
    
    # HC3 structure has 'human_answers' and 'chatgpt_answers' as lists of strings
    for row in dataset:
        for human_ans in row['human_answers']:
            if human_ans.strip():
                texts.append(human_ans)
                labels.append(0) # 0 for Human
        for ai_ans in row['chatgpt_answers']:
            if ai_ans.strip():
                texts.append(ai_ans)
                labels.append(1) # 1 for AI
                
    # Shuffle and sample
    df = pd.DataFrame({'text': texts, 'label': labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if max_samples:
        df = df.head(max_samples)
        
    print(f"Loaded {len(df)} samples ({sum(df['label'] == 0)} Human, {sum(df['label'] == 1)} AI)")
    return df['text'].tolist(), df['label'].tolist()

def main():
    parser = argparse.ArgumentParser(description="Benchmark pre-trained AI text detector")
    parser.add_argument("--model", type=str, default="roberta-base-openai-detector", help="Hugging Face model ID")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to evaluate on")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading model and tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()
    
    texts, true_labels = load_hc3_dataset(split='train', max_samples=args.samples)
    
    predicted_labels = []
    predicted_probs = []
    
    print("Running inference...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size)):
            batch_texts = texts[i:i+args.batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits
            # roberta-base-openai-detector outputs: 0 -> Fake (AI), 1 -> Real (Human)
            # We mapped: 0 -> Human, 1 -> AI
            # So we need to align the labels.
            # Let's check model.config.id2label to be safe.
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            
            # Align predictions with our labels:
            # If model's label 1 is "Real" (Human), and 0 is "Fake" (AI)
            # We want: 0 -> Human, 1 -> AI
            # Let's map model's 1 to our 0, and model's 0 to our 1.
            aligned_preds = [1 if p == 0 else 0 for p in preds]
            predicted_labels.extend(aligned_preds)
            
            # Probabilities for AI class (which is class 0 in this model)
            ai_probs = probs[:, 0].tolist()
            predicted_probs.extend(ai_probs)
            
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
    print(f"F1 Score (AI class): {f1_score(true_labels, predicted_labels):.4f}")
    try:
        roc_auc = roc_auc_score(true_labels, predicted_probs)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    except Exception as e:
        print(f"ROC-AUC Score: N/A ({e})")
        
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=["Human (0)", "AI (1)"]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))

if __name__ == "__main__":
    main()
