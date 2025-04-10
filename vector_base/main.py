# 4. main.py
from data_preprocessor import preprocess_data
from vector_db_manager import YogaVectorDB
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Step 1: Prepare data
    input_csv = "yoga_poses.csv"
    meta_data, documents = preprocess_data(input_csv)
    
    # Step 2: Initialize vector DB
    yoga_db = YogaVectorDB()
    
    # Step 3: Populate database
    yoga_db.populate_db(
        documents=documents,
        metadatas=meta_data.to_dict('records'),
        ids=[str(id) for id in meta_data['AID']]
    )
    
    # Example query
    results = yoga_db.query_poses(
        query_text="Gentle exercises for arthritis relief",
        level="Beginners",
        max_results=3
    )
    
    # Pretty-print results
    print("Search Results:")
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"\nPose Level: {meta['Level']}")
        print(f"Contraindications: {meta['Contraindications'] or 'None'}")
        print(f"Content:\n{doc}\n{'-'*50}")

if __name__ == "__main__":
    main()
