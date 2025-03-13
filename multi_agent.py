import pandas as pd
import re
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

def filter_contraindications(row, contra_list):
    """Check if a yoga pose contains any user contraindications"""
    contra = row["Contraindications"]
    if pd.isna(contra):
        return True
    contra = contra.lower()
    return not any(keyword in contra for keyword in contra_list)

# ===== Multi-Agent Embedding Models =====
st_model = SentenceTransformer('all-MiniLM-L6-v2')
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def get_embeddings(text):
    """Get embeddings from both models"""
    return {
        'st': st_model.encode(text),
        'use': use_model([text]).numpy()[0]  # USE expects list input
    }

# ===== Command Line Arguments =====
parser = argparse.ArgumentParser(description='Multi-Agent Yoga Recommendation System')
parser.add_argument('--health_goal', type=str, required=True,
                   help='Describe your health goals (e.g., "reduce back pain")')
parser.add_argument('--contraindications', type=str, default='',
                   help='Comma-separated contraindications (e.g., "high blood pressure,spinal injury")')
args = parser.parse_args()

# ===== Data Loading & Processing =====
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
df = df[df["Benefits"].str.len() > 5]

# ===== Generate Embeddings =====
print("Generating embeddings with multi-agent system...")
df["embeddings"] = df["Benefits"].apply(get_embeddings)
df["st_embedding"] = df["embeddings"].apply(lambda x: x['st'])
df["use_embedding"] = df["embeddings"].apply(lambda x: x['use'])

# ===== User Input Processing =====
user_goal = args.health_goal.strip()
if not user_goal:
    print("Error: Health goal cannot be empty")
    exit(1)

user_contra = [c.strip().lower() 
              for c in args.contraindications.split(',') 
              if c.strip()]

# Generate user embeddings
user_embeddings = get_embeddings(user_goal)

# ===== Ensemble Similarity Calculation =====
def calculate_combined_similarity(df, user_emb):
    """Calculate combined similarity scores from both models"""
    st_embeddings = np.stack(df["st_embedding"].values)
    use_embeddings = np.stack(df["use_embedding"].values)
    
    st_sim = cosine_similarity(st_embeddings, [user_emb['st']]).flatten()
    use_sim = cosine_similarity(use_embeddings, [user_emb['use']]).flatten()
    
    # Normalize and combine scores
    return (st_sim + use_sim) / 2

df["similarity"] = calculate_combined_similarity(df, user_embeddings)

# ===== Filtering & Recommendations =====
recommended_df = df.sort_values("similarity", ascending=False).head(15)
safe_recommendations = recommended_df[
    recommended_df.apply(lambda row: filter_contraindications(row, user_contra), axis=1)
]

# ===== Save Results =====
if not safe_recommendations.empty:
    safe_recommendations.to_csv("multi_agent_recommendations.csv", 
                               columns=["AName", "Benefits", "Contraindications", "Level", "similarity"], 
                               index=False)
    print(f"Found {len(safe_recommendations)} safe recommendations")
    print("Results saved to multi_agent_recommendations.csv")
else:
    print("No safe recommendations found matching criteria")
