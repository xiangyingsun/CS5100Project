import pandas as pd
import re
import argparse
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

def filter_contraindications(row, contra_list):
    """Check if a yoga pose contains any user contraindications"""
    contra = row["Contraindications"]
    if pd.isna(contra):
        return True
    contra = contra.lower()
    return not any(keyword in contra for keyword in contra_list)

# ===== Command Line Arguments =====
parser = argparse.ArgumentParser(description='Personalized Yoga Asanas Recommendation System')
parser.add_argument('--health_goal', type=str, required=True, 
                   help='Describe your health goals (e.g., "reduce back pain")')
parser.add_argument('--contraindications', type=str, default='',
                   help='Comma-separated contraindications (e.g., "high blood pressure,spinal injury")')
args = parser.parse_args()

# ===== Data Loading & Validation =====
try:
    df = pd.read_csv("yoga_data.csv")
except FileNotFoundError:
    print("Error: yoga_data.csv not found in working directory")
    exit(1)

# Process Benefits column
df["Benefits"] = df["Benefits"].apply(
    lambda x: re.split(r'[.;]', x) if isinstance(x, str) else [])
df = df.explode("Benefits").reset_index(drop=True)
df["Benefits"] = df["Benefits"].str.strip()
df = df[df["Benefits"].str.len() > 5]  # Remove short/empty benefits

# ===== BERT Embeddings =====
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(text):
    inputs = tokenizer(text, 
                      return_tensors="pt",
                      padding=True,
                      truncation=True,
                      max_length=128)
    outputs = model(**inputs)
    # Reduce to 2D using [CLS] token embedding
    return outputs.last_hidden_state[:,0,:].squeeze().detach().numpy() 

# Generate embeddings for all benefits
df["embedding"] = df["Benefits"].apply(lambda x: get_embedding(x))
df.to_csv("asanas_embeddings.csv", index=False)

# ===== User Input Processing =====
if not args.health_goal.strip():
    print("Error: Health goal cannot be empty")
    exit(1)

user_contra = [c.strip().lower() 
              for c in args.contraindications.split(',') 
              if c.strip()]

# Generate user embedding (ensure 2D array)
user_embedding = get_embedding(args.health_goal)

# ===== Vectorized Similarity Calculation =====
all_embeddings = np.stack(df["embedding"].values)
user_embedding = user_embedding.reshape(1, -1)
df["similarity"] = cosine_similarity(all_embeddings, user_embedding).flatten()

# Get top 15 recommendations
recommended_df = df.sort_values("similarity", ascending=False).head(15)

# ===== Contraindications Filtering =====
safe_recommendations = recommended_df[
    recommended_df.apply(lambda row: filter_contraindications(row, user_contra), axis=1)
]

# ===== Save Results =====
if not safe_recommendations.empty:
    safe_recommendations.to_csv("final_recommendations.csv", 
                               columns=["AName", "Benefits", "Contraindications", "Level"], 
                               index=False)
    print("Recommendations saved to final_recommendations.csv")
else:
    print("No safe recommendations found matching criteria")
