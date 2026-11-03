import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import re
from bs4 import BeautifulSoup
import argparse

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags safely
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove excessive whitespace/newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-ascii artifacts
    text = text.encode('ascii', 'ignore').decode()
    
    return text.strip()

def process_hc3(max_samples=None):
    print("Downloading and processing Hello-SimpleAI/HC3 (all.jsonl)...")
    dataset = load_dataset("json", data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl", split="train")
    
    texts = []
    labels = []
    
    for row in dataset:
        # Process Human Answers (Label 0)
        for ans in row.get('human_answers', []):
            texts.append(ans)
            labels.append(0)
            
        # Process AI Answers (Label 1)
        for ans in row.get('chatgpt_answers', []):
            texts.append(ans)
            labels.append(1)
            
    df = pd.DataFrame({'text': texts, 'label': labels})
    print(f"Initial raw rows: {len(df)}")
    
    print("Applying aggressive text cleaning (HTML stripping, normalization)...")
    df['text'] = df['text'].apply(clean_text)
    
    print("Filtering extremely short sequences (< 10 words)...")
    df = df[df['text'].str.split().str.len() >= 10]
    
    # Drop pure duplicates
    df = df.drop_duplicates(subset=['text'])
    
    if max_samples:
        # Sample while trying to preserve class balance if possible, or just sample
        # We will do a stratified sample
        df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), max_samples//2), random_state=42))
        
    df = df.reset_index(drop=True)
    print(f"Final clean rows: {len(df)} (Human: {sum(df['label']==0)}, AI: {sum(df['label']==1)})")
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None, help="Limit total samples for quick testing")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = process_hc3(max_samples=args.max_samples)
    
    # Stratified Split: 80% Train, 10% Val, 10% Test
    print("Performing stratified Train/Val/Test splits...")
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    # Save as high-performance Parquet format
    train_file = os.path.join(args.output_dir, "train.parquet")
    val_file = os.path.join(args.output_dir, "val.parquet")
    test_file = os.path.join(args.output_dir, "test.parquet")
    
    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)
    test_df.to_parquet(test_file, index=False)
    
    print(f"\nPipeline Complete! Saved high-performance datasets to {args.output_dir}/")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    main()
